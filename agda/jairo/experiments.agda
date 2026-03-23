{-# OPTIONS --warn=noUserWarning #-}
module jairo.experiments where
  open import Data.Nat as ℕ using (ℕ)
  open import Data.Bool as B using (if_then_else_)
  open import Data.Float as F using (_+_; _*_; _÷_; e^_; -_; fromℕ; sqrt; _≤ᵇ_) 
    renaming (Float to ℝ)
  open import Data.List as L using (List; []; _∷_; reverse)
  open import Data.List.Properties
  open import Data.List.Relation.Unary.All as All using (All; []; _∷_)
  open import Data.Fin as F using (zero; suc; Fin; combine; remQuot; fromℕ<; 
    inject+; splitAt)
  import Relation.Binary.PropositionalEquality as Eq
  open Eq using (_≡_; refl; trans; sym; cong; cong-app; subst)
  open Eq.≡-Reasoning using (begin_; step-≡; _∎)
  open import Function
  open import Data.Product as Prod using (∃; _,_; _×_; proj₁; proj₂)
  
  infix 4 _⊡_
  _⊡_ = trans

  open import Ar

  module microgpt where

    {- matrix-vector multiplication -}
    linear : Ar (u ⊗ s) ℝ → Ar s ℝ → Ar u ℝ
    linear w x i = sum _+_ 0.0 (zipWith _*_ (nest w i) x)

    {- runs a batch of r linears -}
    mlinear : Ar (r ⊗ (u ⊗ s)) ℝ → Ar s ℝ → Ar (r ⊗ u) ℝ
    mlinear {r} w x = unnest (λ i → linear (nest {r} w i) x)

    {- A transpose? -}
    swap : Ar (u ⊗ s) ℝ → Ar (s ⊗ u) ℝ 
    swap {u} {s} x = unnest {s} λ i j → nest x j i

    {- matrix multiplication but the order is wrong?
    matmulAux : Ar (u ⊗ s) ℝ → Ar (r ⊗ s) ℝ → Ar (r ⊗ u) ℝ
    matmulAux {u} {s} {r} w x = unnest λ (i : P r) → linear w (nest x i)

    matrix multiplication?
    matmul : Ar (u ⊗ s) ℝ → Ar (s ⊗ r) ℝ → Ar (u ⊗ r) ℝ
    matmul {u} {s} {r} w x = swap {r} (matmulAux {u} w (swap {s} x))
    -}

    {- matrix multiplication? -}
    matmul : Ar (u ⊗ s) ℝ → Ar (s ⊗ r) ℝ → Ar (u ⊗ r) ℝ
    matmul {u} {s} {r} w x = unnest {u} (λ i j → linear w (λ k → nest x k j) i)

    {- converts a vector into a probability distribution 
      In microgpt they substract the max value of the vector to
      prevent overflow of exp. Should we do the same?
    -}
    softmax : Ar s ℝ → Ar s ℝ
    softmax {s} x  = let
      exps : Ar s ℝ
      exps = map e^_ x

      total : ℝ
      total = sum _+_ 0.0 exps 

      r = map (_÷ total) exps 
      in r

    {- number of data in an array
      Not sure if this is how I should calculate the length of an array-}
    lenS : S → ℕ
    lenS s = L.foldl ℕ._*_ 1 s

    len : Ar s ℝ → ℕ
    len {s} _ = lenS s

    {- rescales a vector so its values have unit root-mean-square.
      They add 0.0001 in one of the steps. I assume to avoid
      overflow. Should we do the same?
      -}
    rmsnorm : Ar s ℝ → Ar s ℝ
    rmsnorm {s} x = let
      scale : ℝ 
      scale = sqrt ((sum _+_ 0.0 (zipWith _*_ x x)) ÷ (fromℕ (len x))) 

      r = map (_* scale) x
      in r
    
    {- returns the maximum between two floats -}
    maxℝ : ℝ → ℝ → ℝ
    maxℝ a b = if a ≤ᵇ b then b else a

    {- rectified linear unit -}
    relu : Ar s ℝ → Ar s ℝ
    relu = map (maxℝ 0.0)

    {- returns the reverse of a shape.
      I make my own implementation because the og uses fold
      and idk how to define reverseP for that shape. -}
    reverseS : S → S 
    reverseS [] = []
    reverseS (x ∷ xs) = (reverseS xs) ⊗ (x ∷ [])

    {- returns the reverse of a position based onf reverseS -}
    reverseP : P s → P (reverseS s)
    reverseP {[]} [] = []
    reverseP {s ∷ ss} (x ∷ xs) = reverseP xs ++ (x ∷ [])

    {- from reverse to the original.
       I need this to define transpose-}
    unreverseP : P (reverseS s) → P s
    unreverseP {[]} [] = []
    unreverseP {s ∷ ss} x = x₁ ++ x₂ where
      x₁ : P (s ∷ [])
      x₁ = proj₂ (splitP x)

      x₂ : P ss
      x₂ = unreverseP (proj₁ (splitP x))

    {- Transpose of an array? -}
    transpose : Ar s ℝ → Ar (reverseS s) ℝ
    transpose x i = x (unreverseP i)

    {- computes an attention block -}
    attention : {u s r t : S} → Ar (u ⊗ s) ℝ → Ar (r ⊗ s) ℝ → Ar (r ⊗ t) ℝ 
              → Ar (u ⊗ t) ℝ
    attention {u} {s} {r} {t} q k v = let
      {- Are len and swap correct here? -}
      scale : ℝ 
      scale = (sqrt (fromℕ (len v)))

      l : Ar (u ⊗ r) ℝ
      l = map (_÷ scale) (matmul {u} q (swap {r} k))

      w : Ar (u ⊗ t) ℝ
      w = matmul {u} (softmax l) v
      in w

    {- multi headed attention-}
    mattention : {h u s r t : S} 
               → Ar (h ⊗ (u ⊗ s)) ℝ → Ar (h ⊗ (r ⊗ s)) ℝ → Ar (h ⊗ (r ⊗ t)) ℝ 
               → Ar (h ⊗ (u ⊗ t)) ℝ
    mattention {h} {u} q k v = 
      unnest {h} λ i → attention {u} (nest q i) (nest k i) (nest v i)
    
    {- embedding dimension (C) -}
    ED : ℕ
    ED = 16

    {- 
    number of attention heads (H) 
    Must be such that ED/AH is a natural
    -}
    AH : ℕ
    AH = 4

    {- number of layers (N) 
      It is 1 microgpt so perhaps we do not need it? -}
    NL : ℕ
    NL = 1

    {- Dimension of each head -}
    HD : ℕ
    HD = ED ℕ./ AH

    {- sequence length (T) 
       A sequence is a list of tokens.
       In microgpt tokens are letters and sequences names -}
    SL : ℕ
    SL = 16

    {- The feed forward network projects into FDx-}
    FD : ℕ
    FD = 4

    {- I chose this order to make it easy to pass into mattention-}
    IS : S
    IS = (AH ∷ []) ⊗  ((HD ∷ []) ⊗ (SL ∷ []))

    {- assuming a single layer -}
    {- Following gpt2, with differences: 
        layernorm -> rmsnorm
        no biases
        Gelu -> Relu
        -}

    gptLayer : 
        {ah hd sl fd : S} → 
        let is = (ah ⊗ (hd ⊗ sl)) in
          (inp : Ar is ℝ) 
        → (wqkv : Ar (3 ∷ [] ⊗ (is ⊗ is)) ℝ)  
        → (wo : Ar (is ⊗ is) ℝ)
        → (wf1 : Ar ((fd ⊗ is) ⊗ is) ℝ) 
        → (wf2 : Ar (is ⊗ (fd ⊗ is)) ℝ) 
        → Ar is ℝ
    gptLayer {ah} {hd} {sl} {fd} inp wqkv wo wf1 wf2 = let 
      is : S
      is = ah ⊗ (hd ⊗ sl)

      qkv : Ar (3 ∷ [] ⊗ is) ℝ
      qkv = mlinear {3 ∷ []} wqkv (rmsnorm inp)

      q : Ar is ℝ
      q = nest qkv $ zero ∷ []

      k : Ar is ℝ
      k = nest qkv $ suc zero ∷ []

      v : Ar is ℝ
      v = nest qkv $ suc (suc zero) ∷ []

      c₁ : Ar is ℝ
      c₁ = mattention {ah} {hd} q k v

      s₁ : Ar is ℝ
      s₁ = zipWith _+_ (linear wo c₁) inp

      s₂ : Ar (fd ⊗ is) ℝ 
      s₂ = relu $ linear wf1 (rmsnorm s₁)

      c₃ : Ar is ℝ 
      c₃ = linear wf2 s₂

      r = zipWith _+_ c₃ s₁
      in r

    {-
    gptLayer : 
          (inp : Ar IS ℝ) 
        → (wqkv : Ar (3 ∷ [] ⊗ (IS ⊗ IS)) ℝ)  
        → (wo : Ar (IS ⊗ IS) ℝ)
        → (wf1 : Ar ((4 ∷ [] ⊗ IS) ⊗ IS) ℝ) 
        → (wf2 : Ar (IS ⊗ (4 ∷ [] ⊗ IS)) ℝ) 
        → Ar IS ℝ
    gptLayer inp wqkv wo wf1 wf2 = let 
      qkv : Ar (3 ∷ [] ⊗ IS) ℝ
      qkv = mlinear {3 ∷ []} wqkv (rmsnorm inp)

      q : Ar IS ℝ
      q = nest qkv $ zero ∷ []

      k : Ar IS ℝ
      k = nest qkv $ suc zero ∷ []

      v : Ar IS ℝ
      v = nest qkv $ suc (suc zero) ∷ []

      c₁ : Ar IS ℝ
      c₁ = mattention {AH ∷ []} {HD ∷ []} {SL ∷ []} {HD ∷ []} q k v

      s₁ : Ar IS ℝ
      s₁ = zipWith _+_ (linear wo c₁) inp

      s₂ : Ar (4 ∷ [] ⊗ IS) ℝ 
      s₂ = relu $ linear wf1 (rmsnorm s₁)

      c₃ : Ar IS ℝ 
      c₃ = linear wf2 s₂

      r = zipWith _+_ c₃ s₁
      in r
    -}