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
      scale = √ ((sum _+_ 0ᵣ (zipWith _*_ x x)) ÷ (fromℕ (size x)))

      r = map (_* scale) x
      in r

    -- Rectified Linear Unit
    relu : Ar s R → Ar s R
    relu = map (0ᵣ ∨_)

    -- Attention block
    {- I cheat by passing the scale sc as a parameter. It should be such that
     sqrt (size v) =  sc.
     For microgpt sc = 16.
  -}
    attention : {slq slk dk dv : S} → R
              → Ar (slq ⊗ dk) R → Ar (slk ⊗ dk) R → Ar (slk ⊗ dv) R
              → Ar (slq ⊗ dv) R
    attention {slq} {slk} {dk} {dv} sc hq hk hv = let
      l : Ar (slq ⊗ slk) R
      l = map (_÷ sc) (matmul {slq} hq (swap {slk} hk))

      w : Ar (slq ⊗ dv) R
      w = matmul {slq} (softmax l) hv
      in w

    -- Multi-headed attention
    mattention : {nh slq slk dk dv : S} → R
               → Ar (nh ⊗ (slq ⊗ dk)) R → Ar (nh ⊗ (slk ⊗ dk)) R
               → Ar (nh ⊗ (slk ⊗ dv)) R → Ar (nh ⊗ (slq ⊗ dv)) R
    mattention {nh} {slq} {slk} sc q k v =
      unnest {nh} λ i →
        attention {slq} {slk} sc (nest q i) (nest k i) (nest v i)

    -- A single layer forward pass
    layer : {nh sl dh df : S} → (sc : R) →
            let is = nh ⊗ (sl ⊗ dh) in
            (inp : Ar is R)
          → (wa : Ar ((4 ∷ []) ⊗ (is ⊗ is)) R)
          → (wf : Ar ((2 ∷ []) ⊗ ((df ⊗ is) ⊗ is)) R)
          → Ar is R
    layer {nh} {sl} {dh} {df} sc inp wa wf = let
      is = nh ⊗ (sl ⊗ dh)

      ninp = rmsnorm inp

      q : Ar is R
      q = linear (nest wa (zero ∷ [])) ninp

      k : Ar is R
      k = linear (nest wa (suc zero ∷ [])) ninp

      v : Ar is R
      v = linear (nest wa (suc (suc zero) ∷ [])) ninp

      wo : Ar (is ⊗ is) R
      wo = nest wa (suc (suc (suc zero)) ∷ [])

      wf1 : Ar ((df ⊗ is) ⊗ is) R
      wf1 = nest wf (zero ∷ [])

      wf2 : Ar (is ⊗ (df ⊗ is)) R
      wf2 = swap {df ⊗ is} (nest wf (suc zero ∷ []))

      c₁ : Ar is R
      c₁ = mattention {nh} {sl} sc q k v

      s₁ : Ar is R
      s₁ = zipWith _+_ (linear wo c₁) inp

      s₂ : Ar (df ⊗ is) R
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
      3. ReLU instead of GeLU
    1. 2. could be potential be included. 3. is more difficult since
    it uses a probability density function.
      -}
    gpt : {n : ℕ} {nh sl dh df : S} → (sc : R) →
        let is = (nh ⊗ (sl ⊗ dh)) in
          Ar is R
        → Ar (n ∷ [] ⊗ (4 ∷ [] ⊗ (is ⊗ is))) R
        → Ar (n ∷ [] ⊗ (2 ∷ [] ⊗ ((df ⊗ is) ⊗ is))) R
        → Ar is R
    gpt {n} {nh} {hd} {sl} {df} sc inp wa wf =
      sum’ (λ w' inp' → layer {nh} {hd} {df = df} sc inp' (proj₁ w') (proj₂ w'))
        inp λ (i : P (n ∷ [])) → Prod._,_ (nest wa i) (nest wf i)

module Microgpt (real : Real) where
  open Real.Real real
  open GPT real

  NL = 1 ; NH = 4 ; SL = 16  ; DH = 4 ; DF = 4

  sc : R
  sc = √ fromℕ DH

  nh : S ; sl : S ; dh : S ; df : S
  nh = NH ∷ [] ; sl = SL ∷ [] ; dh = DH ∷ [] ; df = DF ∷ []

  is : S
  is = nh ⊗ (sl ⊗ dh)

  microgpt : Ar is R
            → Ar (NL ∷ [] ⊗ (4 ∷ [] ⊗ (is ⊗ is))) R
            → Ar (NL ∷ [] ⊗ (2 ∷ [] ⊗ ((df ⊗ is) ⊗ is))) R
            → Ar is R
  microgpt inp wa wf = gpt {nh = nh} {sl = sl} {df = df} sc inp wa wf