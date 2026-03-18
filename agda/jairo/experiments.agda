{-# OPTIONS --warn=noUserWarning #-}
module jairo.experiments where
  open import Data.Nat as ℕ using (ℕ)
  open import Data.Float as F using (_+_; _*_; _÷_; e^_; -_; fromℕ) renaming (Float to ℝ)
  open import Data.List as L using (List; []; _∷_)
  open import Data.List.Relation.Unary.All as All using (All; []; _∷_)
  open import Data.Fin as F using (zero; suc; Fin; combine; remQuot; fromℕ<; inject+; splitAt)
  import Relation.Binary.PropositionalEquality as Eq
  open Eq using (_≡_; refl; trans; sym; cong; cong-app; subst)
  open Eq.≡-Reasoning using (begin_; step-≡; _∎)
  open import Function
  
  infix 4 _⊡_
  _⊡_ = trans

  open import Ar
  
  module CNN where
    conv : s + p ≈ r → Ar r ℝ → Ar s ℝ → suc p ≈ u → Ar u ℝ
    conv sp a w su = 
      sum (zipWith _+_) (K 0.0) λ i → map (w i *_) (slide i sp a su)

    mconv : ⦃ s + p ≈ r ⦄ → (inp : Ar r ℝ) (w : Ar (u ⊗ s) ℝ) (b : Ar u ℝ)
          → ⦃ suc p ≈ q ⦄ → Ar (u ⊗ q) ℝ
    mconv ⦃ sp ⦄ inp w b ⦃ su ⦄ 
      = unnest λ i → map (b i +_) (conv sp inp (nest w i) su)

    logistic : Ar s ℝ → Ar s ℝ
    logistic = map λ x → 1.0 ÷ (1.0 + e^ (- x))

    avgp₂ : (m n : ℕ) → Ar (m ℕ.* 2 ∷ n ℕ.* 2 ∷ []) ℝ → Ar (m ∷ n ∷ []) ℝ
    avgp₂ m n a = map ((_÷ fromℕ 4) ∘ sum _+_ 0.0) (selb a it)
  
    forward : 
        (inp  :  Ar (28 ∷ 28 ∷ []) ℝ) → (k₁ : Ar (6 ∷ 5 ∷ 5 ∷ []) ℝ)
      → (b₁   :  Ar (6  ∷ []) ℝ)      → (k₂ : Ar (12 ∷ 6 ∷ 5 ∷ 5 ∷ []) ℝ)
      → (b₂   :  Ar (12 ∷ []) ℝ)      → (fc : Ar (10 ∷ 12 ∷ 1 ∷ 4 ∷ 4 ∷ []) ℝ)
      → (b    :  Ar (10 ∷ []) ℝ)      → Ar (10 ∷ 1 ∷ 1 ∷ 1 ∷ 1 ∷ []) ℝ
    forward inp k₁ b₁ k₂ b₂ fc b = let
        c₁ : Ar (6 ∷ 24 ∷ 24 ∷ []) ℝ
        c₁ = logistic $ mconv inp k₁ b₁ 

        s₁ : Ar (6 ∷ 12 ∷ 12 ∷ []) ℝ
        s₁ = unnest {s = 6 ∷ []} $ map (avgp₂ 12 12) (nest c₁)

        c₂ : Ar (12 ∷ 1 ∷ 8 ∷ 8 ∷ []) ℝ
        c₂ = logistic $ mconv  s₁ k₂ b₂ 

        s₂ : Ar (12 ∷ 1 ∷ 4 ∷ 4 ∷ []) ℝ
        s₂ = unnest {s = 12 ∷ 1 ∷ []} $ map (avgp₂ 4 4) (nest c₂)

        r = logistic $ mconv s₂ fc b 
      in r

  module microgpt where

    {- split attention from the beggining, mantaining shapes. -}

    {- matrix-vector multiplication -}

    linear : Ar (u ⊗ s) ℝ → Ar s ℝ → Ar u ℝ
    linear w x i = sum _+_ 0.0 (zipWith _*_ (nest w i) x)

    {-
      linear' : Ar (n ∷ []) ℝ → Ar (m ∷ n ∷ []) ℝ → Ar (m ∷ []) ℝ
      linear' x w i = sum _+_ 0.0 λ j → (nest w i j) * x j    

      test : (x : Ar (n ∷ []) ℝ) → (w : Ar (m ∷ n ∷ []) ℝ) 
         → ∀ i → linear x w i ≡ linear' x w i
      test {n} {m} x w (i ∷ []) = refl -}

    {- converts a vector into a probability distribution -}
    softmax : Ar (n ∷ []) ℝ → Ar (n ∷ []) ℝ

    {- rescales a vector so its values have unit root-mean-square -}
    rmsnorm : Ar s ℝ → Ar s ℝ

    {- computes an attention block -}
    attention : Ar (3 ∷ n ∷ []) ℝ → Ar (n ∷ []) ℝ 

    mattention : Ar (3 ∷ m ∷ n ∷ []) ℝ → Ar (m ∷ n ∷ []) ℝ 

    {- computes a feed forward block -}
    feedForward : Ar (u ⊗ s) ℝ → Ar (s ⊗ u) ℝ → Ar s ℝ → Ar s ℝ
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

    {- sequence length (T) (mattention AH 4 q k v) 
      does not seem to be used in microgpt after tokenization
      perhaps we do not need it? -}
    SL : ℕ
    SL = {!   !}

    IS : S
    IS = AH ∷ HD ∷ []

    {- assuming a single layer -}
    gptLayer : 
          (inp : Ar IS ℝ) 
        → (wqkv : Ar (3 ∷ [] ⊗ (IS ⊗ IS)) ℝ)  
        → (wo : Ar (IS ⊗ IS) ℝ)
        → (wf1 : Ar ((4 ∷ [] ⊗ IS) ⊗ IS) ℝ) 
        → (wf2 : Ar (IS ⊗ (4 ∷ [] ⊗ IS)) ℝ) 
        → Ar IS ℝ
    gptLayer inp wqkv wo wf1 wf2 = let 
      nInp : Ar (AH ∷ HD ∷ []) ℝ
      nInp = rmsnorm inp

      qkv : Ar ((3 ∷ []) ⊗ IS) ℝ
      qkv = linear wqkv nInp

      c₁ : Ar IS ℝ
      c₁ = mattention qkv

      s₁ : Ar IS ℝ
      s₁ = zipWith _+_ (linear wo c₁) inp

      c₂ : Ar IS ℝ
      c₂ = feedForward wf1 wf2 (rmsnorm s₁)

      r = zipWith _+_ c₂ s₁
      in r

      {-
      q : Ar (ED ∷ []) ℝ
      q = linear nInp wq

      k : Ar (ED ∷ []) ℝ
      k = linear nInp wk

      v : Ar (ED ∷ []) ℝ
      v = linear nInp wv
      
      c₁ : Ar (ED ∷ []) ℝ
      c₁ = mattention AH 4 q k v

      s₁ : Ar (ED ∷ []) ℝ
      s₁ = zipWith _+_ (linear c₁ wo) inp
        
      c₂ : Ar (ED ∷ []) ℝ
      c₂ = feedForward wf1 wf2 (rmsnorm s₁)

      r = zipWith _+_ c₂ s₁
      -}

      
