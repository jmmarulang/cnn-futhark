{-# OPTIONS  --backtracking-instance-search #-}
-- {-# OPTIONS --warn=noUserWarning #-}

module _ where
module _ where
  open import Ar hiding (sum; slide; backslide; imapb; selb)
  open import Relation.Binary.PropositionalEquality
  open import Data.Product
  open import Data.Nat using (ℕ; zero; suc; _≟_)
  open import Data.List as L
  open import Data.List.Properties as L
  open import Relation.Nullary
  open import Data.Maybe
  open import Function
  open import Lang
  open import Ar
  open import LangEq

  open WkSub

  -- replace x with y in e, if x is any subexpression in e.
  replace : (e : E Γ is) (x y : E Γ ip) → E Γ is
  replace e x y with e-eq? e x
  ... | just (refl , refl) = y
  replace (var v) x y | nothing = var v
  replace zero x y | nothing = zero
  replace one x y | nothing = one
  replace (imaps e) x y | nothing = imaps (replace e (x ↑) (y ↑))
  replace (sels e e₁) x y | nothing = sels (replace e x y) (replace e₁ x y)
  replace (imap e) x y | nothing = imap (replace e (x ↑) (y ↑))
  replace (sel e e₁) x y | nothing = sel (replace e x y) (replace e₁ x y)
  replace (E.imapb x₁ e) x y | nothing = E.imapb x₁ (replace e (x ↑) (y ↑))
  replace (E.selb x₁ e e₁) x y | nothing = E.selb x₁ (replace e x y) (replace e₁ x y)
  replace (E.sum e) x y | nothing = E.sum (replace e (x ↑) (y ↑))
  replace (zero-but e e₁ e₂) x y | nothing = zero-but (replace e x y) (replace e₁ x y) (replace e₂ x y)
  replace (E.slide e x₁ e₁ x₂) x y | nothing = E.slide (replace e x y) x₁ (replace e₁ x y) x₂
  replace (E.backslide e e₁ x₁ x₂) x y | nothing = E.backslide (replace e x y) (replace e₁ x y) x₁ x₂
  -- replace (logistic e) x y | nothing = logistic (replace e x y)
  replace (bin x₁ e e₁) x y | nothing = bin x₁ (replace e x y) (replace e₁ x y)
  replace (scaledown x₁ e) x y | nothing = scaledown x₁ (replace e x y)
  -- replace (minus e) x y | nothing = minus (replace e x y)
  replace (let′ e e₁) x y | nothing = let′ (replace e x y) (replace e₁ (x ↑) (y ↑))
  -- Jairo made
  replace (un x₁ e) x y | nothing = un x₁ (replace e x y)
  replace (maximum e) x y | nothing = maximum (replace e (x ↑) (y ↑))

  replace-let : (e : E Γ is) → E Γ is
  replace-let (let′ e e₁) = let e' = (replace-let e) in
    let′ e' (replace (replace-let e₁) (e' ↑) (var v₀)) -- is this correct?
  replace-let (var x) = var x
  replace-let 𝟘 = 𝟘
  replace-let 𝟙 = 𝟙
  replace-let (imaps e) = imaps (replace-let e)
  replace-let (sels e e₁) = sels (replace-let e) (replace-let e₁)
  replace-let (imap e) = imap (replace-let e)
  replace-let (sel e e₁) = sel (replace-let e) (replace-let e₁)
  replace-let (E.imapb x e) = E.imapb x (replace-let e)
  replace-let (E.selb x e e₁) = E.selb x (replace-let e) (replace-let e₁)
  replace-let (E.sum e) = E.sum (replace-let e)
  replace-let (zero-but e e₁ e₂) =
    zero-but (replace-let e) (replace-let e₁) (replace-let e₂)
  replace-let (E.slide e x e₁ x₁) =
    E.slide (replace-let e) x (replace-let e₁) x₁
  replace-let (E.backslide e e₁ x x₁) =
    E.backslide (replace-let e) (replace-let e₁) x x₁
  replace-let (bin x e e₁) = bin x (replace-let e) (replace-let e₁)
  replace-let (scaledown x e) = scaledown x (replace-let e)
  replace-let (un x e) = un x (replace-let e)
  replace-let (maximum e) = maximum (replace-let e)

  -- inline : E Γ is → E Γ is
  -- inline e = norm-lets (inline' e) where
  --   inline' : E Γ is → E Γ is
  --   inline' (var x) = var x
  --   inline' 𝟘 = 𝟘
  --   inline' 𝟙 = 𝟙
  --   inline' (imaps e) = imaps (inline' e)
  --   inline' (sels e e₁) = sels (inline' e) (inline' e₁)
  --   inline' (imap e) = imap (inline e)
  --   inline' (sel e e₁) = sel (inline' e) (inline' e₁)
  --   inline' (imapb x e) = E.imapb x (inline' e)
  --   inline' (selb x e e₁) = E.selb x (inline' e) (inline' e₁)
  --   inline' (sum e) = E.sum (inline' e)
  --   inline' (zero-but e e₁ e₂) = (zero-but (inline' e) (inline' e₁) (inline' e₂))
  --   inline' (slide e x e₁ x₁) = E.slide (inline' e) x (inline' e₁) x₁
  --   inline' (backslide e e₁ x x₁) = E.backslide (inline' e) (inline' e₁) x x₁
  --   inline' (bin x e e₁) = bin x (inline' e) (inline' e₁)
  --   inline' (scaledown x e) = scaledown x (inline' e)
  --   inline' (un x e) = un x (inline' e)
  --   inline' (maximum e) = maximum (inline' e)
  --   inline' (let′ e e₁) with a ← (inline' e₁) | count-uses a v₀ | e
  --   ... | 0 | _ = sub a (sub-id ▹ (inline' e))
  --   ... | _ | var b = sub a (sub-id ▹ (var b))
  --   ... | _ | zero = sub a (sub-id ▹ zero)
  --   ... | _ | one = sub a (sub-id ▹ one)
  --   -- ... | 1 | (imap b) = let t = sel b (var {! here  !}) in {!   !}
  --   -- ... | 1 | _ = sub a (sub-id ▹ (inline' e))
  --   ... | _ | _ = let′ (inline' e) a

module Test where
  open import Data.List

  open import Lang
  open Syntax

  ex₁ : E _ _
  ex₁ = Lcon (ar [] ∷ []) (ar []) ε
        λ a → Let x := (Let y := a ⊞ a In (y) ⊞ (y)) In x
  -- let′ (let′ (var v₀ ⊞ var v₀) (var v₀ ⊞ var v₀)) (var v₀)

  ex-repl = replace ex₁ (var v₀ ⊞ var v₀) one
  -- let′ (let′ 𝟙 (var v₀ ⊞ var v₀)) (var v₀)
