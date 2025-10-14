## HOAS-like Wrappers

<!--
```
{-# OPTIONS  --backtracking-instance-search #-}
open import present

module _ where
module Syntax where
 open Lang
 open import Data.List as L using (List; []; _∷_)
 open A hiding (sum)
 open WkSub
```
-->
<div style="font-size:80%;">
```
 data Prefix : (Γ Δ : Ctx) → Set where instance
  zero : Prefix Γ Γ
  suc  : ⦃ Prefix Γ Δ ⦄ → Prefix Γ (Δ ▹ is)

 GE : Ctx → IS → Set
 GE Γ is = ∀ {Δ} → ⦃ Prefix Γ Δ ⦄ → E Δ is

 ⟨_⟩ : E Γ is → GE Γ is

 Imap : (GE (Γ ▹ ix s) (ix s) → E (Γ ▹ ix s) (ar p))
      → E Γ (ar (s ⊗ p))
 Imap f = imap (f ⟨ var v₀ ⟩)

 _ : E ε _ 
 _ = Imap {s = ι 5}
     λ i → Imap {s = ι 5}
           λ j → sels (sel one j) i
```
</div>

<!--
```
 GV : Ctx → IS → Set
 GV Γ is = ∀ {Δ} → ⦃ p : Prefix Γ Δ ⦄ → is ∈ Δ
 
 ⟨_⟩ᵛ : is ∈ Γ → GV Γ is

 prefix-⊆ : Prefix Γ Δ → Γ ⊆ Δ
 prefix-⊆ zero         = ⊆-eq
 prefix-⊆ (suc ⦃ p ⦄)  = skip (prefix-⊆ p)
 
 ⟨_⟩ t {Δ} ⦃ p ⦄ = wk (prefix-⊆ p) t
 ⟨_⟩ᵛ v ⦃ p ⦄ = wkv (prefix-⊆ p) v
 Sum : ∀ {Γ}
      → (GE (Γ ▹ ix s) (ix s) → E (Γ ▹ ix s) (ar p))
      → E Γ (ar p)
 Sum f = sum (f λ {Δ} ⦃ p ⦄ → var ⟨ v₀ ⟩ᵛ)

 Imaps : ∀ {Γ}
       → (GE (Γ ▹ ix s) (ix s) → E (Γ ▹ ix s) (ar []))
       → E Γ (ar s)
 Imaps f = imaps (f λ {Δ} ⦃ p ⦄ → var ⟨ v₀ ⟩ᵛ)

 Imapb : ∀ {Γ}
       → s * p ≈ q 
       → (GE (Γ ▹ ix s) (ix s) → E (Γ ▹ ix s) (ar p)) 
       → E Γ (ar q)
 Imapb p f = imapb p (f λ {Δ} ⦃ p ⦄ → var ⟨ v₀ ⟩ᵛ)

 Let-syntax : E Γ (ar s) → (GE (Γ ▹ (ar s)) (ar s) → E (Γ ▹ (ar s)) (ar p)) → E Γ (ar p)
 Let-syntax x f = let′ x (f (var ⟨ v₀ ⟩ᵛ))
 syntax Let-syntax e (λ x → e') = Let x := e In e'

 _ : E ε (ar [])
 _ = Let x := one In Let y := x ⊞ one In (x ⊞ y) ⊠ x  

 infixl 3 Let-syntax

 ext : Ctx → List IS → Ctx
 ext Γ []      = Γ
 ext Γ (x ∷ l) = ext (Γ ▹ x) l

 lfun : (l : List IS)  (Γ : Ctx) (is : IS) → Set
 lvar : ∀ l → is ∈ Γ → GE (ext Γ l) is

 -- Turn the list of IS into the following function:
 --   l = [a, b, c]
 --   X = X
 --   Γ = Γ
 --   ----------------------------
 --   GE Γ a → GE Γ b → GE Γ c → X
 lfunh : (l : List IS) (X : Set) (Γ : Ctx) → Set
 lfunh [] X Γ = X
 lfunh (a ∷ l) X Γ = GE Γ a → lfunh l X Γ

 -- Diagonalise lfunh:
 --   l = [a, b]
 --   Γ = Γ
 --   is = is
 --   ---------------------------------------------
 --   GE (ext Γ l) a → GE (ext Γ l) → E (ext Γ l) is
 lfun l Γ τ = lfunh l (E (ext Γ l) τ) (ext Γ l)
 lvar [] v = var ⟨ v ⟩ᵛ
 lvar (x ∷ l) v = lvar l (vₛ v)

 Lcon : ∀ l is Γ → (f : lfun l Γ is) → E (ext Γ l) is
 Lcon []      is Γ f  = f
 Lcon (x ∷ l) is Γ f  = Lcon l is (Γ ▹ x) (f (lvar l v₀))

 _ : E _ _
 _ = Lcon (ar (ι 5) ∷ ar (5 ∷ 5 ∷ []) ∷ []) (ar []) ε
     λ a b → Sum λ i → sels a i ⊞ sels (sel b i) i
```
-->

## CNN in E

<!--
```
module Primitives where
 open import Data.List as L using (List; []; _∷_)
 open import Data.Nat as ℕ using (ℕ; zero; suc)
 open import Function using (_$_; it; _∋_)
 open import Relation.Binary.PropositionalEquality
 open A hiding (slide; conv; mconv; avgp₂; logistic)
 open Syntax
 open WkSub
 open Lang


 conv : E Γ (ar r) → ⦃ s + p ≈ r ⦄ → E Γ (ar s) → ⦃ suc p ≈ u ⦄ → E Γ (ar u)
 conv f ⦃ s+p ⦄ g ⦃ ss ⦄ = Sum λ i → (slide i s+p ⟨ f ⟩ ss) ⊠ Imaps λ j → sels ⟨ g ⟩ i

 mconv : ⦃ s + p ≈ r ⦄ → (inp : E Γ (ar r)) (ws : E Γ (ar (u ⊗ s)))
         (bᵥ : E Γ (ar u)) → ⦃ suc p ≈ w ⦄ → E Γ (ar (u ⊗ w))
 mconv ⦃ sp ⦄ inp wᵥ bᵥ ⦃ su ⦄ = Imap λ i → conv ⟨ inp ⟩ (sel ⟨ wᵥ ⟩ i) ⊞ Imaps λ _ → sels ⟨ bᵥ ⟩ i

 avgp₂ : ∀ m n → (a : E Γ (ar (m ℕ.* 2 ∷ n ℕ.* 2 ∷ []))) → E Γ (ar (m ∷ n ∷ []))
 avgp₂ m n a = Imaps λ i → scaledown 4 $ Sum λ j → sels (selb it ⟨ a ⟩ i) j

 sqerr : (r o : E Γ (ar [])) → E Γ (ar [])
 sqerr r o = scaledown 2 ((r ⊟ o) ⊠ (r ⊟ o))

 meansqerr : (r o : E Γ (ar s)) → E Γ (ar [])
 meansqerr r o = Sum λ i → sqerr (sels ⟨ r ⟩ i) (sels ⟨ o ⟩ i) 
```
-->

<div style="font-size:80%;">
```
 cnn : E _ _
 cnn = 
  Lcon (ar (28 ∷ 28 ∷ []) ∷ ar (6 ∷ 5 ∷ 5 ∷ [])
        ∷ ar (6 ∷ []) ∷ ar (12 ∷ 6 ∷ 5 ∷ 5 ∷ [])
        ∷ ar (12 ∷ []) ∷ ar (10 ∷ 12 ∷ 1 ∷ 4 ∷ 4 ∷ [])
        ∷ ar (10 ∷ []) ∷ ar (10 ∷ 1 ∷ 1 ∷ 1 ∷ 1 ∷ [])
        ∷ [])
       (ar []) ε
  λ inp k₁ b₁ k₂ b₂ fc b target → 
  Let c₁₁ := mconv inp k₁ b₁  In
  Let c₁  := logistic c₁₁ In
  Let s₁  := (Imap {s = 6 ∷ []}
              λ i → avgp₂ 12 12 (sel c₁ i)) In
  Let c₂₁ := mconv s₁ k₂ b₂ In
  Let c₂  := logistic c₂₁ In
  Let s₂  := (Imap {s = 12 ∷ 1 ∷ []}
              λ i → avgp₂ 4 4 (sel c₂ i)) In
  Let o₁  := mconv s₂ fc b In
  Let o   := logistic o₁ In
  Let e   := meansqerr target o In
  e
```
</div>
