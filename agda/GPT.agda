open import Data.Nat as ℕ using (ℕ)
open import Data.Bool as B using (if_then_else_)
open import Data.List as L using (List; []; _∷_; reverse)
open import Data.List.Properties
open import Data.List.Relation.Unary.All as All using (All; []; _∷_)
open import Data.Fin as Fin using (zero; suc; Fin; combine; remQuot; fromℕ<;
    inject+; splitAt)
import Relation.Binary.PropositionalEquality as Eq
open Eq using (_≡_; refl; trans; sym; cong; cong-app; subst)
open Eq.≡-Reasoning using (begin_; step-≡; _∎)
open import Function
open import Data.Product as Prod using (∃; _,_; _×_; proj₁; proj₂)

open import Ar
open import Real

module _ where
module GPT (real : Real) where
    open Real.Real real

    -- Matrix-vector multiplication
    linear : Ar (u ⊗ s) R → Ar s R → Ar u R
    linear w x i = sum _+_ 0ᵣ (zipWith _*_ (nest w i) x)

    -- Matrix multiplication
    matmul : Ar (u ⊗ s) R → Ar (s ⊗ r) R → Ar (u ⊗ r) R
    matmul {u} {s} {r} w x = unnest {u} (λ i j → linear w (λ k → nest x k j) i)

    -- Converts a vector into a probability distribution
    {-
        In microgpt they substract the max value of the vector to
        prevent overflow of exp. Should we do the same?
    -}
    softmax : Ar s R → Ar s R
    softmax {s} x  = let
      exps : Ar s R
      exps = map e^_ x

      total : R
      total = sum _+_ 0ᵣ exps

      r = map (_÷ total) exps
      in r

    rmsnorm : Ar s R → Ar s R
    rmsnorm {s} x = let
      scale : R
      scale = sqrt ((sum _+_ 0ᵣ (zipWith _*_ x x)) ÷ (fromℕ (size x)))

      r = map (_* scale) x
      in r

    -- Rectified Linear Unit
    relu : Ar s R → Ar s R
    relu = map (0ᵣ ∧_)

    -- Attention block
    attention : {u s r t : S} → Ar (u ⊗ s) R → Ar (r ⊗ s) R → Ar (r ⊗ t) R
              → Ar (u ⊗ t) R
    attention {u} {s} {r} {t} q k v = let
      scale : R
      scale = (sqrt fromℕ (size v))

      l : Ar (u ⊗ r) R
      l = map (_÷ scale) (matmul {u} q (swap {r} k))

      w : Ar (u ⊗ t) R
      w = matmul {u} (softmax l) v
      in w

    -- Multi-headed attention
    mattention : {h u s r t : S}
               → Ar (h ⊗ (u ⊗ s)) R → Ar (h ⊗ (r ⊗ s)) R → Ar (h ⊗ (r ⊗ t)) R
               → Ar (h ⊗ (u ⊗ t)) R
    mattention {h} {u} q k v =
      unnest {h} λ i → attention {u} (nest q i) (nest k i) (nest v i)

    -- A single layer forward pass
    layer :
        {ah hd sl fd : S} →
        let is = (ah ⊗ (hd ⊗ sl)) in
          (inp : Ar is R)
        → (wa : Ar (4 ∷ [] ⊗ (is ⊗ is)) R)
        → (wf : Ar (2 ∷ [] ⊗ ((fd ⊗ is) ⊗ is)) R)
        → Ar is R
    layer {ah} {hd} {sl} {fd} inp wa wf = let
      is = ah ⊗ (hd ⊗ sl)

      ninp = rmsnorm inp

      q : Ar is R
      q = linear (nest wa (zero ∷ [])) ninp

      k : Ar is R
      k = linear (nest wa (suc zero ∷ [])) ninp

      v : Ar is R
      v = linear (nest wa (suc (suc zero) ∷ [])) ninp

      wo : Ar (is ⊗ is) R
      wo = nest wa (suc (suc (suc zero)) ∷ [])

      wf1 : Ar ((fd ⊗ is) ⊗ is) R
      wf1 = nest wf (zero ∷ [])

      wf2 : Ar (is ⊗ (fd ⊗ is)) R
      wf2 = swap {fd ⊗ is} (nest wf (suc zero ∷ []))

      c₁ : Ar is R
      c₁ = mattention {ah} {hd} q k v

      s₁ : Ar is R
      s₁ = zipWith _+_ (linear wo c₁) inp

      s₂ : Ar (fd ⊗ is) R
      s₂ = relu $ linear wf1 (rmsnorm s₁)

      c₃ : Ar is R
      c₃ = linear wf2 s₂

      r = zipWith _+_ c₃ s₁
      in r

    -- GPT2 forward pass
    {-
   With the following simplifications:
      1. RMSNorm instead of LayerNorm
      2. no biases
      3. ReLU instaed of GeLU
    1. 2. could be potential be included. 3. is more difficult since
    it uses a probability density function.
      -}
    gpt : {n : ℕ} {ah hd sl fd : S} →
        let is = (ah ⊗ (hd ⊗ sl)) in
          Ar is R
        → Ar (n ∷ [] ⊗ (4 ∷ [] ⊗ (is ⊗ is))) R
        → Ar (n ∷ [] ⊗ (2 ∷ [] ⊗ ((fd ⊗ is) ⊗ is))) R
        → Ar is R
    gpt {n} {ah} {hd} {sl} {fd} inp wa wf =
      sum’ (λ w' inp' → layer {ah} {hd} {fd = fd} inp' (proj₁ w') (proj₂ w'))
        inp λ (i : P (n ∷ [])) → Prod._,_ (nest wa i) (nest wf i)

module Microgpt (real : Real) where
  open Real.Real real
  open GPT real
  -- Embedding dimension
  ED : ℕ
  ED = 16

  -- Number of attention heads
  {- Must be such that ED/AH is a natural. Should we add this as a condition? -}
  AH : ℕ
  AH = 4

  -- Number of layers
  NL : ℕ
  NL = 1

  -- Size of each head
  HD : ℕ
  HD = ED ℕ./ AH

  -- Sequence length
  {-
  A sequence is a list of tokens.
  In microgpt tokens are letters and sequences are names.
  -}
  SL : ℕ
  SL = 16

  --  Input shape
  {- This order makes it easy to pass into mattention -}
  I : S
  I = (AH ∷ []) ⊗  ((HD ∷ []) ⊗ (SL ∷ []))

  -- The feedforward network projects into (FD x IS) x IS
  FD : ℕ
  FD = 4

  microgpt :
    (inp : Ar I R) →
    Ar (NL ∷ [] ⊗ (4 ∷ [] ⊗ (I ⊗ I))) R
    → Ar (NL ∷ [] ⊗ (2 ∷ [] ⊗ ((FD ∷ [] ⊗ I) ⊗ I))) R →
    Ar I R
  microgpt = gpt {ah = AH ∷ []} {hd = HD ∷ []} {fd = FD ∷ []}


