"""Top-K sparse autoencoder implementation and training.

Implementation details:
1. Matryoshka loss: average the loss computed with increasing prefixes of neurons.
2. Auxiliary-K reconstruction: revive dead neurons.
3. Multi-K reconstruction: use a slightly less sparse reconstruction with lower weight.
4. Batch Top-K sparsity (optional via flag): during training select top-(K·B) activations across the batch
   and maintain a learned activation threshold used at inference to preserve expected sparsity.

Our implementation draws from:
- Bussmann et al. (2025) https://github.com/bartbussmann/matryoshka_sae/blob/main/sae.py (for Matryoshka loss)
- O'Neill et al. (2024) https://github.com/Christine8888/saerch/blob/main/saerch/topk_sae.py (for top-K with dead neuron revival)
"""

import os
import pickle
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification



class SepSae(nn.Module):
    """Two independent sparse autoencoders for two document collections.

    Documents are embedded independently with the same frozen text embedder.
    Each embedding is then encoded and reconstructed by its own SAE, allowing
    the two streams to use different numbers of active neurons.
    """

    def __init__(
        self,
        m_total_neurons: int,
        k1: int,
        k2: int,
        *,
        embed_name: str = "nomic-ai/modernbert-embed-base",
        embedder: Optional[nn.Module] = None,
        tokenizer=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **sae_kwargs,
    ) -> None:
        super().__init__()
        if k1 < 1 or k2 < 1:
            raise ValueError("k1 and k2 must be positive")

        if embedder is None or tokenizer is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if embedder is None:
                embedder = AutoModelForSequenceClassification.from_pretrained(
                    embed_name
                )
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(embed_name)

        self.embedder = embedder.eval()
        self.tokenizer = tokenizer
        self.embed_name = embed_name
        self.device = torch.device(device)
        self.embedder.to(self.device)
        for parameter in self.embedder.parameters():
            parameter.requires_grad_(False)

        self.embed_dim = self._get_embed_dim(self.embedder)
        self.sae1 = SparseAutoencoder(
            input_dim=self.embed_dim,
            m_total_neurons=m_total_neurons,
            k_active_neurons=k1,
            device=device,
            **sae_kwargs,
        )
        self.sae2 = SparseAutoencoder(
            input_dim=self.embed_dim,
            m_total_neurons=m_total_neurons,
            k_active_neurons=k2,
            device=device,
            **sae_kwargs,
        )

    @staticmethod
    def _get_embed_dim(embedder: nn.Module) -> int:
        config = getattr(embedder, "config", None)
        embed_dim = getattr(config, "hidden_size", None)
        if embed_dim is None:
            embed_dim = getattr(config, "hidden_dim", None)
        if embed_dim is None:
            raise ValueError("embedder.config must define hidden_size or hidden_dim")
        return int(embed_dim)

    def _embed(self, batch) -> torch.Tensor:
        if hasattr(self.embedder, "base_model"):
            outputs = self.embedder.base_model(**batch)
        else:
            outputs = self.embedder(**batch)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0]
        if isinstance(outputs, dict) and "last_hidden_state" in outputs:
            return outputs["last_hidden_state"][:, 0]
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        raise ValueError("embedder output must provide last_hidden_state or pooler_output")

    @torch.inference_mode()
    def preprocess(
        self,
        documents1: list[str] | str,
        documents2: list[str] | str,
        batch_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed each document collection independently.

        The two collections may contain different numbers of documents. The
        returned tensors are on the model device and can be passed to
        :meth:`forward`.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.embedder.eval()

        def embed_documents(documents: list[str] | str) -> torch.Tensor:
            if isinstance(documents, str):
                documents = [documents]
            if not documents:
                return torch.empty((0, self.embed_dim), device=self.device)
            embeddings = []
            indices = list(range(0, len(documents), batch_size))
            for start in tqdm(indices, desc='Computing embeddings'):
                batch = self.tokenizer(
                    documents[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                batch = {
                    key: value.to(self.device) for key, value in batch.items()
                }
                embeddings.append(self._embed(batch))
            return torch.cat(embeddings, dim=0)

        return embed_documents(documents1), embed_documents(documents2)

    def forward(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, Dict], Tuple[torch.Tensor, Dict]]:
        """Reconstruct two batches of embeddings with their corresponding SAE."""
        if embeddings1.shape[-1] != self.embed_dim or embeddings2.shape[-1] != self.embed_dim:
            raise ValueError(f"both embeddings must have final dimension {self.embed_dim}")
        return self.sae1(embeddings1), self.sae2(embeddings2)

    def compute_loss(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        outputs1: Tuple[torch.Tensor, Dict],
        outputs2: Tuple[torch.Tensor, Dict],
        aux_coef: float = 1 / 32,
        multi_coef: float = 0.0,
    ) -> torch.Tensor:
        """Return the mean reconstruction loss for both embedding streams."""
        recon1, info1 = outputs1
        recon2, info2 = outputs2
        loss1 = self.sae1.compute_loss(
            embeddings1, recon1, info1, aux_coef, multi_coef
        )
        loss2 = self.sae2.compute_loss(
            embeddings2, recon2, info2, aux_coef, multi_coef
        )
        return (loss1 + loss2) / 2

    def fit(
        self,
        X_train1: torch.Tensor,
        X_train2: torch.Tensor,
        X_val1: Optional[torch.Tensor] = None,
        X_val2: Optional[torch.Tensor] = None,
        batch_size: int = 512,
        learning_rate: float = 5e-4,
        n_epochs: int = 200,
        aux_coef: float = 1 / 32,
        multi_coef: float = 0.0,
        patience: int = 5,
        show_progress: bool = True,
        clip_grad: float = 1.0,
    ) -> Dict:
        """Train both sparse autoencoders on their respective embeddings."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(X_train1) == 0 or len(X_train2) == 0:
            raise ValueError("training embeddings must not be empty")
        if (X_val1 is None) != (X_val2 is None):
            raise ValueError("X_val1 and X_val2 must be provided together")
        if X_val1 is not None and (len(X_val1) == 0 or len(X_val2) == 0):
            raise ValueError("validation embeddings must not be empty")

        train_loaders = (
            DataLoader(TensorDataset(X_train1), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(X_train2), batch_size=batch_size, shuffle=True),
        )
        val_loaders = None
        if X_val1 is not None and X_val2 is not None:
            val_loaders = (
                DataLoader(TensorDataset(X_val1), batch_size=batch_size),
                DataLoader(TensorDataset(X_val2), batch_size=batch_size),
            )

        self.sae1.initialize_weights_(X_train1.to(self.device))
        self.sae2.initialize_weights_(X_train2.to(self.device))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        history = {"train_loss": [], "val_loss": [], "dead_neuron_ratio": []}
        best_val_loss = float("inf")
        patience_counter = 0

        def paired_batches(loaders):
            iterators = [iter(loader) for loader in loaders]
            remaining = max(len(loader) for loader in loaders)
            for index in range(remaining):
                batches = []
                for loader_index, iterator in enumerate(iterators):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterators[loader_index] = iter(loaders[loader_index])
                        batch = next(iterators[loader_index])
                    batches.append(batch[0])
                yield batches

        iterator = tqdm(range(n_epochs)) if show_progress else range(n_epochs)
        for epoch in iterator:
            self.train()
            self.embedder.eval()
            train_losses = []
            for batch1, batch2 in paired_batches(train_loaders):
                batch1 = batch1.to(self.device)
                batch2 = batch2.to(self.device)
                outputs1, outputs2 = self(batch1, batch2)
                loss = self.compute_loss(
                    batch1, batch2, outputs1, outputs2, aux_coef, multi_coef
                )

                optimizer.zero_grad()
                loss.backward()
                self.sae1.adjust_decoder_gradient_()
                self.sae2.adjust_decoder_gradient_()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                optimizer.step()
                self.sae1.normalize_decoder_()
                self.sae2.normalize_decoder_()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            history["train_loss"].append(avg_train_loss)
            dead_ratio = float(
                torch.cat(
                    (self.sae1.steps_since_activation, self.sae2.steps_since_activation)
                )
                .gt(self.sae1.dead_neuron_threshold_steps)
                .float()
                .mean()
                .item()
            )
            history["dead_neuron_ratio"].append(dead_ratio)

            if val_loaders is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for batch1, batch2 in paired_batches(val_loaders):
                        batch1 = batch1.to(self.device)
                        batch2 = batch2.to(self.device)
                        outputs1, outputs2 = self(batch1, batch2)
                        val_losses.append(
                            self.compute_loss(
                                batch1,
                                batch2,
                                outputs1,
                                outputs2,
                                aux_coef,
                                multi_coef,
                            ).item()
                        )

                avg_val_loss = float(np.mean(val_losses))
                history["val_loss"].append(avg_val_loss)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if show_progress:
                iterator.set_postfix(
                    train_loss=f"{avg_train_loss:.4f}",
                    val_loss=(
                        f"{history['val_loss'][-1]:.4f}"
                        if val_loaders is not None
                        else "N/A"
                    ),
                    dead_ratio=f"{dead_ratio:.3f}",
                )

        return history


# ----------------------------------------------------------------------------
# Sparse Autoencoder with optional Matryoshka loss
# ----------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        m_total_neurons: int,
        k_active_neurons: int,
        *,
        aux_k: Optional[int] = None,
        multi_k: Optional[int] = None,
        dead_neuron_threshold_steps: int = 256,
        prefix_lengths: Optional[List[int]] = None,
        use_batch_topk: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Create a top-K sparse autoencoder.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input representations.
        m_total_neurons : int
            Total number of neurons (SAE features).
        k_active_neurons : int
            Number of active neurons selected per example.
        aux_k : int | None, optional
            Upper bound on the number of dead neurons to try for auxiliary residual prediction.
            The default value, set during initialization, is `2 * k_active_neurons`.
            The default coefficient on the auxiliary loss is 1/32 (see `compute_loss`).
        multi_k : int | None, optional
            How many neurons to use for the less-sparse multi-K reconstruction
            term.  The default weight on this loss is 0, and the default value
            is None (i.e. no multi-K reconstruction).  If using, a recommend starting 
            value is `4 * k_active_neurons`.
        dead_neuron_threshold_steps : int, optional
            Steps of non-activation after which a neuron counts as *dead*.
        prefix_lengths : list[int] | None, optional
            If given (e.g. `[16, 64]`), activates *Matryoshka* loss: the first
            prefix has 16 neurons, the second 64, etc.  If *None*, all
            M neurons are treated equally.
        """

        super().__init__()
        self.input_dim = input_dim
        self.m_total_neurons = m_total_neurons
        self.k_active_neurons = k_active_neurons
        self.use_batch_topk = use_batch_topk

        # Fallback defaults ---------------------------------------------------
        self.aux_k = (
            min(2 * k_active_neurons, m_total_neurons) if aux_k is None else aux_k
        )
        self.multi_k = multi_k
        self.dead_neuron_threshold_steps = dead_neuron_threshold_steps

        # Matryoshka prefixes as full lengths --------------------------------
        self.prefix_lengths = prefix_lengths
        if self.prefix_lengths is not None:
            assert (
                self.prefix_lengths[-1] == m_total_neurons
            ), "Last prefix length must equal m_total_neurons"
            assert all(
                x > y for x, y in zip(self.prefix_lengths[1:], self.prefix_lengths[:-1])
            ), "Each prefix length must be greater than the previous one"

        # weight initialization --------------------------------------------------------------
        self.encoder = nn.Linear(input_dim, m_total_neurons, bias=False)
        self.decoder = nn.Linear(m_total_neurons, input_dim, bias=False)

        self.input_bias = nn.Parameter(torch.zeros(input_dim))
        self.neuron_bias = nn.Parameter(torch.zeros(m_total_neurons))

        # dead-neuron bookkeeping --------------------------------------------
        self.steps_since_activation = torch.zeros(
            m_total_neurons, dtype=torch.long, device=device
        )

        # Batch-Top-K threshold buffer (used only when use_batch_topk=True)
        self.register_buffer("threshold", torch.tensor(0.0))

        self.device = device
        self.to(self.device)

    # ---------------------------------------------------------------------
    # Forward pass (agnostic to Matryoshka configuration)
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # W_enc(x - b_pre) + b_enc
        x = x - self.input_bias
        pre_act = self.encoder(x) + self.neuron_bias

        # -------- Shared: produce sparse activations 'activ' -------------
        topk_idx = topk_vals = None  # reported only for hard Top-K
        if not self.use_batch_topk:
            # main Top-K ---------------------------------------------------------
            local_vals, local_idx = torch.topk(pre_act, self.k_active_neurons, dim=-1)
            local_vals = F.relu(local_vals)
            activ = torch.zeros_like(pre_act)
            activ.scatter_(-1, local_idx, local_vals)
            topk_idx, topk_vals = local_idx, local_vals
        else:
            acts = F.relu(pre_act)
            if self.training:
                batch_size = acts.shape[0]
                flat_acts = acts.flatten()
                k_total = min(self.k_active_neurons * batch_size, flat_acts.numel())
                vals_flat, idx_flat = torch.topk(flat_acts, k_total, dim=-1)
                activ_flat = torch.zeros_like(flat_acts)
                activ_flat.scatter_(0, idx_flat, vals_flat)
                activ = activ_flat.view_as(acts)
                # update learned threshold towards smallest positive activation
                self._update_threshold_(activ)
            else:
                # inference-time thresholding to match expected sparsity
                activ = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))

        # -------------------- Shared: multi-K ----------------------------
        if self.multi_k is not None:
            multik_vals, multik_idx = torch.topk(pre_act, self.multi_k, dim=-1)
            multik_vals = F.relu(multik_vals)
            multik_activ = torch.zeros_like(pre_act)
            multik_activ.scatter_(-1, multik_idx, multik_vals)
            multik_recon = self.decoder(multik_activ) + self.input_bias
        else:
            multik_recon = None

        # -------------------- Shared: dead-neuron tracking ---------------
        self.steps_since_activation += 1
        fired = (activ.sum(dim=0) > 0).nonzero(as_tuple=False).squeeze(-1)
        if fired.numel() > 0:
            self.steps_since_activation.index_fill_(0, fired, 0)

        # -------------------- Shared: reconstruction ---------------------
        recon = self.decoder(activ) + self.input_bias

        # -------------------- Shared: aux-K ------------------------------
        aux_idx = aux_vals = None
        if self.aux_k is not None:
            dead_mask = (self.steps_since_activation > self.dead_neuron_threshold_steps).float()
            dead_pre_act = pre_act * dead_mask
            aux_vals, aux_idx = torch.topk(dead_pre_act, self.aux_k, dim=-1)
            aux_vals = F.relu(aux_vals)

        info = {
            "activations": activ,  # needed for Matryoshka slices
            "topk_indices": topk_idx,
            "topk_values": topk_vals,
            "multik_reconstruction": multik_recon,
            "aux_indices": aux_idx,
            "aux_values": aux_vals,
        }
        return recon, info

    # ------------------------------------------------------------------
    # Loss with optional Matryoshka terms
    # ------------------------------------------------------------------
    @staticmethod
    def _normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        baseline_mse = F.mse_loss(target.mean(dim=0, keepdim=True).expand_as(target), target)
        return mse / baseline_mse
    
    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        info: Dict[str, torch.Tensor],
        aux_coef: float,
        multi_coef: float,
    ) -> torch.Tensor:
        """Return total loss (Matryoshka L2 + optional multi-K + aux).

        If `len(prefix_lengths)==1` there is no Matryoshka nesting.
        Otherwise we average the L2 of every prefix reconstruction as in
        Bussmann et al. (2025).

        multiK / auxK implemented as in O'Neill et al. (2024).
        """

        activ = info["activations"]
        # main L2 -----------------------------------------------------------
        if self.prefix_lengths is None or len(self.prefix_lengths) == 1:
            main_l2 = self._normalized_mse(recon, x)
        else:
            l2_terms = []
            dec_weight = self.decoder.weight  # (input_dim, m_total_neurons)
            for end in self.prefix_lengths:
                # activ[:, :end] is (batchsize, end);  dec_weight[:, :end] is (input_dim, end)
                prefix_recon = activ[:, :end] @ dec_weight[:, :end].t() + self.input_bias
                l2_terms.append(self._normalized_mse(prefix_recon, x))
            main_l2 = torch.stack(l2_terms).mean()

        # multi-K term ------------------------------------------------------
        if multi_coef != 0 and info["multik_reconstruction"] is not None:
            main_l2 = main_l2 + multi_coef * self._normalized_mse(
                info["multik_reconstruction"], x
            )

        # aux-K term --------------------------------------------------------
        if self.aux_k is not None and info["aux_indices"] is not None:
            err = x - recon.detach()
            aux_act = torch.zeros_like(activ)
            aux_act.scatter_(-1, info["aux_indices"], info["aux_values"])
            err_recon = self.decoder(aux_act)
            aux_loss = self._normalized_mse(err_recon, err)
            return main_l2 + aux_coef * aux_loss
        else:
            return main_l2

    # ------------------------------------------------------------------
    # Utility helpers for training
    # ------------------------------------------------------------------
    def normalize_decoder_(self):
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))

    def adjust_decoder_gradient_(self):
        if self.decoder.weight.grad is not None:
            with torch.no_grad():
                proj = (self.decoder.weight * self.decoder.weight.grad).sum(dim=0, keepdim=True)
                self.decoder.weight.grad.sub_(proj * self.decoder.weight)

    def initialize_weights_(self, data_sample: torch.Tensor):
        self.input_bias.data = torch.median(data_sample, dim=0).values
        nn.init.xavier_uniform_(self.decoder.weight)
        self.normalize_decoder_()
        self.encoder.weight.data = self.decoder.weight.t().clone()
        nn.init.zeros_(self.neuron_bias)

    @torch.no_grad()
    def _update_threshold_(self, activ: torch.Tensor, lr: float = 1e-2):
        """EMA update towards the smallest positive activation in the current batch."""
        pos = activ > 0
        if pos.any():
            min_pos = activ[pos].min()
            self.threshold.mul_(1 - lr).add_(lr * min_pos)
        
    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        config = {
            "input_dim": self.input_dim,
            "m_total_neurons": self.m_total_neurons,
            "k_active_neurons": self.k_active_neurons,
            "aux_k": self.aux_k,
            "multi_k": self.multi_k,
            "dead_neuron_threshold_steps": self.dead_neuron_threshold_steps,
            "prefix_lengths": self.prefix_lengths,
            "use_batch_topk": self.use_batch_topk,
        }
        torch.save({"config": config, "state_dict": self.state_dict()}, save_path, pickle_module=pickle)
        print(f"Saved model to {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        batch_size: int = 512,
        learning_rate: float = 5e-4,
        n_epochs: int = 200,
        aux_coef: float = 1 / 32,
        multi_coef: float = 0.0,
        patience: int = 5,
        show_progress: bool = True,
        clip_grad: float = 1.0
    ) -> Dict:
        """Train the sparse autoencoder on input data."""
        train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size) if X_val is not None else None
        
        # Initialize from batch of data
        self.initialize_weights_(X_train.to(self.device))
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop setup
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'dead_neuron_ratio': []}
        
        # Training loop
        iterator = tqdm(range(n_epochs)) if show_progress else range(n_epochs)
        for epoch in iterator:
            self.train()
            train_losses = []
            
            for batch_x, in train_loader:
                batch_x = batch_x.to(self.device)
                recon, info = self(batch_x)
                loss = self.compute_loss(batch_x, recon, info, aux_coef, multi_coef)
                
                optimizer.zero_grad()
                loss.backward()
                self.adjust_decoder_gradient_()
                
                # Apply gradient clipping
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                
                optimizer.step()
                self.normalize_decoder_()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Track dead neurons
            dead_ratio = (self.steps_since_activation > self.dead_neuron_threshold_steps).float().mean().item()
            history['dead_neuron_ratio'].append(dead_ratio)
            
            # Validation
            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_x, in val_loader:
                        batch_x = batch_x.to(self.device)
                        recon, info = self(batch_x)
                        val_loss = self.compute_loss(batch_x, recon, info, aux_coef, multi_coef)
                        val_losses.append(val_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            # Update progress bar
            if show_progress:
                postfix = {
                    'train_loss': f'{avg_train_loss:.4f}',
                    'val_loss': f'{avg_val_loss:.4f}' if val_loader else 'N/A',
                    'dead_ratio': f'{dead_ratio:.3f}'
                }
                if self.use_batch_topk:
                    postfix['threshold'] = f'{self.threshold.item():.2e}'
                iterator.set_postfix(postfix)
        
        # Save final model
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            filename = get_sae_checkpoint_name(self.m_total_neurons, self.k_active_neurons, self.prefix_lengths)
            self.save(os.path.join(save_dir, filename))
            
        return history
    
    # ------------------------------------------------------------------
    # Compute activations with batched SAE inference
    # ------------------------------------------------------------------
    def get_activations(self, inputs, batch_size=16384, show_progress=True):
        """Get sparse activations for input data with batching to prevent CUDA OOM.
        
        Args:
            inputs: Input data as numpy array or torch tensor
            batch_size: Number of samples per batch (default: 16384)
        
        Returns:
            Numpy array of activations
        """
        self.eval()

        if isinstance(inputs, list):
            inputs = torch.tensor(inputs, dtype=torch.float)
        elif isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        elif not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a list, numpy array, or torch tensor")
        if not inputs.dtype == torch.float:
            inputs = inputs.float()
        
        num_samples = inputs.shape[0]
        all_activations = []
        with torch.no_grad():
            if show_progress:
                iterator = tqdm(range(0, num_samples, batch_size), desc=f"Computing activations (batchsize={batch_size})")
            else:
                iterator = range(0, num_samples, batch_size)
                
            for i in iterator:
                batch = inputs[i:i+batch_size]
                batch = batch.to(self.device)
                _, info = self(batch)
                batch_activations = info['activations']
                all_activations.append(batch_activations.cpu())
        
        return torch.cat(all_activations, dim=0).numpy()

# -----------------------------------------------------------------------------
# Additional utils
# -----------------------------------------------------------------------------
def get_sae_checkpoint_name(m_total_neurons, k_active_neurons, prefix_lengths=None):
    if prefix_lengths is None:
        return f'SAE_M={m_total_neurons}_K={k_active_neurons}.pt'
    else:
        prefix_str = "-".join(str(g) for g in prefix_lengths)
        return f'SAE_matryoshka_M={m_total_neurons}_K={k_active_neurons}_prefixes={prefix_str}.pt'

def load_model(path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> SparseAutoencoder:
    ckpt = torch.load(path, pickle_module=pickle)
    model = SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded model from {path} onto device {model.device}")
    return model

class CrossSae(SparseAutoencoder):
    def __init__(self,
                embed_name = "nomic-ai/modernbert-embed-base",
                 **kwargs):
        embedder = AutoModelForSequenceClassification.from_pretrained(embed_name).eval()
        embed_dim = embedder.config.hidden_size
        print(f"embed_dim={embed_dim}")
        super().__init__(input_dim=2 * embed_dim, **kwargs)
        self.embed_dim = embed_dim
        self.embedder: torch.nn.Module = embedder
        self.tokenizer = AutoTokenizer.from_pretrained(embed_name)
        for w in self.embedder.parameters():
            w.requires_grad_ = False

        self.film_gate = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim, bias=True),
            torch.nn.Tanh()
        )
        self.film_bias = torch.nn.Linear(embed_dim, embed_dim)
        # self.post_tran = torch.nn.Linear(embed_dim, 2 * embed_dim)

    @torch.inference_mode
    def preprocess(self, targets: list[str], text: list[str], batch_size: int = 16):
        if len(targets) != len(text):
            raise ValueError("targets and text must have the same length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        device = next(self.embedder.parameters()).device
        if not targets:
            return torch.empty((0, 2 * self.embed_dim), device=device)

        def embed_batch(items: list[str]):
            tokenized = self.tokenizer(
                items,
                padding=False,
                truncation=True,
                return_attention_mask=True,
            )
            sorted_indices = sorted(
                range(len(items)),
                key=lambda index: len(tokenized["input_ids"][index]),
            )
            dataloader = DataLoader(
                sorted_indices,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda indices: self.tokenizer.pad(
                    {
                        key: [tokenized[key][index] for index in indices]
                        for key in tokenized
                    },
                    return_tensors="pt",
                ),
                pin_memory=device.type == "cuda",
            )

            sorted_embeddings = []
            for batch in tqdm(dataloader, desc="Embedding batches"):
                sorted_embeddings.extend(self._embed(
                    batch.to(device, non_blocking=device.type == "cuda")
                ))

            embeddings = [None] * len(items)
            for index, embedding in zip(sorted_indices, sorted_embeddings):
                embeddings[index] = embedding
            return torch.stack(embeddings)

        text_embeddings = embed_batch(list(text))
        target_embeddings = embed_batch(list(targets))
        return torch.cat((target_embeddings, text_embeddings), dim=-1).detach().cpu()

    def _embed(self, doc_ids):
        return self.embedder.base_model(**doc_ids).last_hidden_state[:, 0]

    def forward(self, x: torch.Tensor):
        A = x[..., :self.embed_dim]
        B = x[..., self.embed_dim:]
        mixed = self.film_gate(A) * B + self.film_bias(A)
        recon, info = super().forward( torch.cat([A, mixed], dim=1) )
        return recon, info
