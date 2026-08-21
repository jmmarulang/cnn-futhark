-- {-# OPTIONS --warn=noUserWarning #-}
{-# OPTIONS --allow-unsolved-metas #-}
open import Data.Product
open import Data.Unit
open import Data.Empty
open import Data.Nat using (ℕ; zero; suc)
open import Data.List as L using (List; []; _∷_)
open import Data.List.Relation.Unary.All as All using (All; []; _∷_)
open import Relation.Binary.PropositionalEquality
open import Relation.Nullary
open import Function
open import Data.Nat.Properties using (_≟_)
open import Data.Product.Properties

open import Ar
open import Lang
open import LangEq
open import Real

-- Jairo Made
open import Data.Product as Prod hiding (_<*>_)
open import Data.List.Properties

module Opt (r : Real) (rp : RealProp r) where

  open Real.Real r
  open RealProp rp

  open import Eval r rp
  open ZeroBut
  open WkSub hiding (_∙ˢ_)
  open import Data.Maybe

  ∷-inj₂ : (n ∷ s ≡ n ∷ p) → s ≡ p
  ∷-inj₂ refl = refl

  ++-inj₂ : (s L.++ p ≡ s L.++ q) → p ≡ q
  ++-inj₂ {[]} eq = eq
  ++-inj₂ {x ∷ s} eq = ++-inj₂ (∷-inj₂ eq)

  let-out-stren : ∀ {p r q is}
    → ((E (Γ ▹ ar q ▹ is) (ar p)) → (E (Γ ▹ ar q) (ar r)))
    → ((E (Γ ▹ is) (ar p)) → (E Γ (ar r)))
    → (E (Γ ▹ is) (ar q)) → (E (Γ ▹ is ▹ ar q) (ar p)) → (E Γ (ar r))
  let-out-stren f g a b with (stren-∃ a v₀)
  ... | just (a' , _) = let′ a' (f (sub b sub-swap))
  ... | nothing = g (let′ a b)

  let-out-step : ∀ {is p r} → (∀ {Δ} → (E (Δ ▹ is) (ar p)) → (E Δ (ar r)))
    → (E (Γ ▹ is) (ar p)) → (E Γ (ar r))
  let-out-step f (let′ a b) = let-out-stren f f a b
  let-out-step f e = f e

  let-out : E Γ is → E Γ is
  let-out (imaps e) = let-out-step imaps (let-out e)
  let-out (imap e) = let-out-step imap (let-out e)
  let-out (imapb x e) = let-out-step (E.imapb x) (let-out e)
  let-out (sum e) = let-out-step E.sum (let-out e)
  -- let-out (let′ e (let′ a b)) =
  --   let e' = (let-out e) in
  --   let a' = (let-out a) in
  --   let b' = (let-out b) in
  --   let-out-stren (λ x → let′ (e' ↑) x) (λ x → let′ e' x) a' b'
  let-out (let′ e e₁) = let′ (let-out e) (let-out e₁)
  let-out (sels (let′ a b) e₁) = let′ a (sels b (e₁ ↑))
  let-out (sels e e₁) = sels (let-out e) (let-out e₁)
  let-out (sel e e₁) = sel (let-out e) (let-out e₁)
  let-out (selb x e e₁) = E.selb x (let-out e) (let-out e₁)
  let-out (zero-but e e₁ (let′ a b)) = let′ a (zero-but (e ↑) (e₁ ↑) b)
  let-out (zero-but e e₁ e₂) = zero-but (let-out e) (let-out e₁) (let-out e₂)
  let-out (bin x (let′ a b) e₁) = let′ (let-out a) (bin x (let-out b) ((let-out e₁) ↑))
  let-out (bin x e e₁) with (isLet e₁)
  ... | yes (r , a , b , refl) = let′ (let-out a) (bin x ((let-out e) ↑) (let-out b))
  ... | no _ = bin x (let-out e) (let-out e₁)
  let-out (scaledown x e) = scaledown x (let-out e)
  let-out (un x e) = un x (let-out e)
  let-out (slide e x e₁ x₁) = E.slide (let-out e) x (let-out e₁) x₁
  let-out (backslide e e₁ x x₁) = E.backslide (let-out e) (let-out e₁) x x₁
  let-out e = e

  -- TODO : Incomplete
  sum-in : E Γ is → E Γ is
  sum-in (E.sum (a ⊠ b)) with
    a' ← (sum-in a) | b' ← (sum-in b) | (stren a' v₀) | (stren (sum-in b) v₀)
  ... | just c | _ = c ⊠ E.sum b'
  ... | _ | just d = (E.sum a') ⊠ d
  ... | _ | _ = E.sum (a' ⊠ b')
  sum-in (E.sum (⊟ e)) = ⊟ E.sum (sum-in e)

  sum-in (E.sum e) = E.sum (sum-in e)
  sum-in (imaps e) = imaps (sum-in e)
  sum-in (imap e) = imap (sum-in e)
  sum-in (imapb x e) = E.imapb x (sum-in e)
  sum-in (sels e e₁) = sels (sum-in e) (sum-in e₁)
  sum-in (sel e e₁) = sel (sum-in e) (sum-in e₁)
  sum-in (E.selb x e e₁) = E.selb x (sum-in e) (sum-in e₁)
  sum-in (zero-but e e₁ e₂) = zero-but e e₁ (sum-in e₂)
  sum-in (E.slide e x e₁ x₁) = E.slide (sum-in e) x (sum-in e₁) x₁
  sum-in (E.backslide e e₁ x x₁) = E.backslide (sum-in e) (sum-in e₁) x x₁
  sum-in (bin x e e₁) = bin x (sum-in e) (sum-in e₁)
  sum-in (scaledown x e) = scaledown x (sum-in e)
  sum-in (let′ e e₁) = let′ (sum-in e) (sum-in e₁)
  sum-in (un x e) = un x (sum-in e)
  sum-in e = e

  -- selx-in-step : ∀ {s p r} → (E Γ (ar s) → E Γ (ix p) → E Γ (ar r))
  --   → E Γ (ar s) → E Γ (ix p) → E Γ (ar r)
  -- selx-in-step {s = s} {p = p} {r = r} f (imaps e) i with (s ≟ˢ p) | (r ≟ˢ unit)
  -- ... | yes refl | yes refl = sub e (sub-id ▹ i)
  -- ... | _ | _ = f (imaps e) i
  -- selx-in-step f (imap e) i = {!   !}
  -- selx-in-step f (sel e e₁) i = {!   !}
  -- selx-in-step f (E.imapb x e) i = {!   !}
  -- selx-in-step f (E.selb x e e₁) i = {!   !}
  -- selx-in-step f (E.sum e) i = {!   !}
  -- selx-in-step f (zero-but e e₁ e₂) i = {!   !}
  -- selx-in-step f (E.slide e x e₁ x₁) i = {!   !}
  -- selx-in-step f (E.backslide e e₁ x x₁) i = {!   !}
  -- selx-in-step f (bin x e e₁) i = {!   !}
  -- selx-in-step f (scaledown x e) i = {!   !}
  -- selx-in-step f (let′ e e₁) i = {!   !}
  -- selx-in-step f (un x e) i = {!   !}
  -- selx-in-step f (maximum e) i = {!   !}
  -- selx-in-step f (sels e e₁) i = {!   !}
  -- selx-in-step f e i = f e i

  -- selx-in : E Γ is → E Γ is
  -- selx-in (sels (imaps e) i) = sub (selx-in e) (sub-id ▹ i)
  -- -- selx-in (sels (sels e e₁) i) = {!   !}
  -- selx-in (sels (imap e) i) = sub {!  selx-in e !} {!   !}
  -- selx-in (sels (sel e e₁) i) = {!   !}
  -- selx-in (sels (E.imapb x e) i) = {!   !}
  -- selx-in (sels (E.selb x e e₁) i) = {!   !}
  -- selx-in (sels (E.sum e) i) = {!   !}
  -- selx-in (sels (zero-but e e₁ e₂) i) = {!   !}
  -- selx-in (sels (E.slide e x e₁ x₁) i) = {!   !}
  -- selx-in (sels (E.backslide e e₁ x x₁) i) = {!   !}
  -- selx-in (sels (bin x e e₁) i) = {!   !}
  -- selx-in (sels (scaledown x e) i) = {!   !}
  -- selx-in (sels (let′ e e₁) i) = {!   !}
  -- selx-in (sels (un x e) i) = {!   !}
  -- selx-in (sels (maximum e) i) = {!   !}
  -- selx-in (sels e i) = sels (selx-in e) i

  -- selx-in (sel e i) = {!   !}

  -- selx-in (E.selb x e i) = {!   !}

  -- selx-in (let′ e e₁) = let′ (selx-in e) (selx-in e₁) -- Do something here?
  -- selx-in (imaps e) = imaps (selx-in e)
  -- selx-in (imap e) = imap (selx-in e)
  -- selx-in (E.imapb x e) = E.imapb x (selx-in e)
  -- selx-in (E.sum e) = E.sum (selx-in e)
  -- selx-in (maximum e) = maximum (selx-in e)
  -- selx-in (zero-but e e₁ e₂) = zero-but (selx-in e) (selx-in e₁) (selx-in e₂)
  -- selx-in (E.slide e x e₁ x₁) = E.slide (selx-in e) x (selx-in e₁) x₁
  -- selx-in (E.backslide e e₁ x x₁) = E.backslide (selx-in e) (selx-in e₁) x x₁
  -- selx-in (bin x e e₁) = bin x (selx-in e) (selx-in e₁)
  -- selx-in (scaledown x e) = scaledown x (selx-in e)
  -- selx-in (un x e) = un x (selx-in e)
  -- selx-in e = e

  opt : (e : E Γ is) → ∃ λ e′ → (e ≈ᵉ e′)
  opt (var x) = var x , reflᵉ (var x)
  opt zero = zero , reflᵉ zero
  opt one = one , reflᵉ one
  -- Jairo made
  opt (imaps {s = s} e) with opt e
  -- ... | c , p with isLet c
  -- ... | yes (r , a , b , refl) = let′ (imap a) (imaps
  --   (sub b ((skeep (sdrop sub-id)) ▹ sel (var v₁) (var v₀)))) , foo
  --   where
  --   foo : _
  --   foo ρ i = (p _ [])
  --     ∙ sym (eval-sub b _ _ _
  --     ∙ eval-cong b
  --       (sub-env-wks (sdrop sub-id) (skip ⊆-eq) _ ▹ refl ▹ λ _ → refl) []
  --     ∙ eval-cong b (sub-env-sdrop sub-id ▹ refl ▹ λ _ → refl) []
  --     ∙ eval-cong b (sub-env-id ▹ refl ▹ λ _ → refl) []
  --     ∙ eval-cong b (wk-env-id ▹ refl ▹ λ _ → refl) []
  --     ∙ eval-cong b (reflᶜ ▹ refl ▹ foo') [])
  --     where
  --     foo' : _
  --     foo' l =
  --       cong (eval a _) (splitP-proj₂ {i = i})
  --       ∙ cong (λ x → eval a (_ , x) _) (splitP-proj₁ {i = i})

  -- opt (imaps e) | t , p | _ = imaps t , λ ρ i → p (ρ , i) []

  opt (imaps e) | t , p = imaps t , λ ρ i → p (ρ , i) []

  opt (sels e e₁) with opt e | opt e₁
  ... | (zero , p)        | (i , q) = zero , λ ρ i → p ρ _
  ... | (one , p)         | (i , q) = one , λ ρ i → p ρ _
  ... | (imaps e₂ , p)    | (i , q) = sub e₂ (sub-id ▹ i)
                          , λ {ρ [] → p ρ (eval e₁ ρ)
                                    ∙ cong (λ t → eval e₂ (ρ , t) []) (q ρ)
                                    ∙ sym (eval-sub e₂ ρ (sub-id ▹ i) []
                                           ∙ eval-cong e₂ (sub-env-id ▹ refl) [])}
  ... | (a ⊞ b , p) | (i , q) = (sels a i) ⊞ (sels b i)
                              , λ ρ j → p ρ (eval e₁ ρ)
                                        ∙ cong₂ _+_ (cong (eval a ρ) (q ρ)) (cong (eval b ρ) (q ρ))
  ... | (a ⊠ b , p) | (i , q) = (sels a i) ⊠ (sels b i)
                              , λ ρ j → p ρ (eval e₁ ρ)
                                        ∙ cong₂ _*_ (cong (eval a ρ) (q ρ)) (cong (eval b ρ) (q ρ))
  -- ... | sum e | i = sum (selₛ e (wk here i))
  ... | (sum e , p) | (i , q) = E.sum (sels e (i ↑))
                              , λ {ρ [] → p ρ (eval e₁ ρ)
                                          ∙ sum-inv _+_ (fromℕ 0) {eval e ∘ (ρ ,_)} (eval e₁ ρ)
                                          ∙ sym (sum-inv _+_ (fromℕ 0)
                                                         {λ i₁ i₂ → eval e (ρ , i₁)
                                                                           (eval (wk (skip ⊆-eq) i)
                                                                                 (ρ , i₁))} []
                                                 ∙ Ar.sum-cong _+_ (fromℕ 0)
                                                   {λ j → eval e (ρ , j) (eval (wk (skip ⊆-eq) i) (ρ , j))}
                                                   λ j → cong (eval e (ρ , j))
                                                         (eval-wk (skip ⊆-eq) i (ρ , j)
                                                          ∙ eval-cong i wk-env-id
                                                          ∙ sym (q ρ) ))  }
  -- ... | zero-but i j a | k = zero-but i j (selₛ a k)
  ... | zero-but {Γ = Γ} i j a , p | (k , q) = zero-but i j (sels a k)
                                     , go
          where
            go : (ρ : ⟦ Γ ⟧ᶜ) → ∀ u → eval e ρ (eval e₁ ρ) ≡ eval (zero-but i j (sels a k)) ρ u
            go ρ u with eval i ρ ≟ₚ eval j ρ | p ρ (eval e₁ ρ)
            ... | yes _ | p′ = p′ ∙ cong (eval a ρ) (q ρ)
            ... | no _ | p′ = p′
  ... | (⊟ e₂ , p) | (i , q) = (⊟ (sels e₂ i)) , λ ρ j → (p _ _) ∙ cong (λ x → - (eval e₂ ρ x)) (q _)
  ... | (𝟙/ e₂ , p) | (i , q) = (𝟙/ (sels e₂ i)) , λ ρ j → (p _ _) ∙ cong (λ x → fromℕ 1 ÷  (eval e₂ ρ x)) (q _)

  opt (sels e e₁) | (a , p) | (i , q) = sels a i , λ ρ j → p ρ (eval e₁ ρ) ∙ cong (eval a ρ) (q ρ)

-- Jairo made
  opt (imap {s = s} {p = p} e) with opt e
  ... | t , pf with isLet t
  ... | yes (r , a , b , refl) with (stren-∃ a v₀)
  ... | just (c , eq) = (let′ c (imap (sub b sub-swap))) , foo where
    foo : _
    foo ρ i = let j , k = splitP {s} i .proj₁ , splitP {s} i .proj₂ in
      (pf _ k)
      ∙ sym (eval-sub b _ _ _
      ∙ eval-cong b (sub-env-wks (sdrop sub-id) (skip ⊆-eq) _ ▹ refl ▹ λ _ → refl) k
      ∙ eval-cong b (sub-env-sdrop sub-id ▹ refl ▹ λ _ → refl) k
      ∙ eval-cong b (sub-env-id ▹ refl ▹ λ _ → refl) k
      ∙ eval-cong b (wk-env-id ▹ refl ▹ λ _ → refl) k
      ∙ sym (eval-cong b (reflᶜ ▹ refl ▹ λ l → cong (λ x → eval x _ _) eq) k
      ∙ eval-cong b (reflᶜ ▹ refl ▹ λ l → eval-wk (skip ⊆-eq) c (ρ , j) l) k
      ∙ eval-cong b (reflᶜ ▹ refl ▹ eval-cong c wk-env-id) k
      ))
  ... | nothing = imap t , λ ρ i → pf (ρ , splitP i .proj₁) (splitP i .proj₂)
  opt (imap {s = s} e) | t , pf | _  = imap t , λ ρ i → pf (ρ , splitP i .proj₁) (splitP i .proj₂)

  -- ... | (let′ {s = r} {p = .p} a b) , q =
  --   let′ (imap a) (imap {s = s} {p = p}
  --     (sub b (skeep (sdrop sub-id) ▹ sel (var v₁) (var v₀)))) , foo where
  --   foo : _
  --   foo ρ i = let j , k = splitP {s} i .proj₁ , splitP {s} i .proj₂ in
  --     (q _ k)
  --     ∙ sym (eval-sub b _ _ _
  --     ∙ eval-cong b
  --       (sub-env-wks (sdrop sub-id) (skip ⊆-eq) _ ▹ refl ▹ λ _ → refl) k
  --     ∙ eval-cong b (sub-env-sdrop sub-id ▹ refl ▹ λ _ → refl) k
  --     ∙ eval-cong b (sub-env-id ▹ refl ▹ λ _ → refl) k
  --     ∙ eval-cong b (wk-env-id ▹ refl ▹ λ _ → refl) k
  --     ∙ eval-cong b (reflᶜ ▹ refl ▹ foo') k)
  --     where
  --     foo' : _
  --     foo' l = let j , k = splitP {s} i .proj₁ , splitP {s} i .proj₂ in
  --       cong (eval a _) (splitP-proj₂ {i = j})
  --       ∙ cong (λ x → eval a (_ , x) _) (splitP-proj₁ {i = j})

  -- ... | _ | t , p  = imap t , λ ρ i → p (ρ , splitP i .proj₁) (splitP i .proj₂)

    --let t , p = opt e in imap t , λ ρ i → p (ρ , splitP i .proj₁) (splitP i .proj₂)
  opt (sel {s = s}{p} e e₁) with opt e | opt e₁
  -- ... | zero | i = zero
  ... | a , pf | i , q with isZero a
  ... | yes refl = zero , λ ρ j → pf ρ (eval e₁ ρ ++ j)
  ... | no _ with isOne a
  -- ... | one | i = one
  ... | yes refl = one , λ ρ j → pf ρ (eval e₁ ρ ++ j)
  -- ... | imap e | i = sub here e i
  -- NOTE: This case looks complicated because our definition of Lang uses
  --       _++_ on lists.  The problem is that if we are selecting at shape
  --       (s ++ p) with index (i : P p), and the optimised expression happens
  --       to be imap of shape (s′ ++ p′), it is not guaranteed that
  --       s ≡ s′ and p ≡ p′.  For now we assume that the shapes are static
  --       so we check them.  If we want to generalise this, we need to introduce
  --       symbolic shapes and operations on them which should live in a separate
  --       environment.
  ... | no _ with isImap a
  ... | yes (s′ , p′ , spq , u , eq) with s ≟ˢ s′
  ... | no _
        -- If shapes do not match, give up.  In principle we can handle cases
        -- when s ⊑ s′, but we do not see these cases in practice.
        = sel a i , λ ρ j → pf ρ (eval e₁ ρ ++ j) ∙ cong (eval a ρ) (cong (_++ j) (q ρ))
  ... | yes refl with (++-inj₂ {s = s} spq)
  ... | refl rewrite spq | eq = sub u (sub-id ▹ i)
                              , go
        where go : (ρ : ⟦ _ ⟧ᶜ) (j : P p′) → eval e ρ (eval e₁ ρ ++ j) ≡ eval (sub u (sub-id ▹ _)) ρ _
              go ρ j rewrite q ρ  = pf ρ (eval i ρ ++ j)
                                     ∙ sym (eval-sub u ρ (sub-id ▹ i) j
                                            ∙ eval-cong u (sub-env-id ▹ (sym $ splitP-proj₁ {j = j})) j
                                            ∙ cong (eval u _) (sym $ splitP-proj₂ {i = eval i ρ}))
  -- Jairo Made
  opt (sel {Γ = Γ} {s = s} {p} e e₁) | a , pf | i , q | no _ | no _ | no _ with isZeroBut a
  ... | yes (s , j , k , e₂ , refl) = zero-but {Γ = Γ} j k (sel e₂ i) , go
       where
       go : (ρ : ⟦ Γ ⟧ᶜ) → ∀ u → eval e ρ (eval e₁ ρ ++ u) ≡ eval (zero-but j k (sel e₂ i)) ρ u
       go ρ u with eval j ρ ≟ₚ eval k ρ | pf ρ (eval e₁ ρ ++ u)
       ... | yes _ | p′ = p′ ∙ cong (λ x → eval e₂ ρ (x ++ u)) (q ρ)
       ... | no _ | p′ = p′

  opt (sel {s = s} {p} e e₁) | a , pf | i , q | no _ | no _ | no _ | no _ with isLet a
  ... | yes (r , c , d , refl) = let′ c (sel d (i ↑)) , foo where
    -- let′ c (sel d (wk (skip ⊆-eq) i)) , foo where
    foo : _
    foo ρ j =
      pf ρ _
      ∙ sym (cong (λ x → eval d _ (x ++ j)) (eval-wk (skip ⊆-eq) i _
      ∙ eval-cong i wk-env-id
      ∙ sym (q _)))

  opt (sel {s = s} {p} e e₁) | a , pf | i , q | no _ | no _ | no _ | no _ | no _
  -- ... | a | i = sel a i
    = sel a i , λ ρ j → pf ρ (eval e₁ ρ ++ j) ∙ cong (eval a ρ) (cong (_++ j) (q ρ))

  -- opt (sel {s = s} {p} e e₁) | a , pf | i , q | no _ | no _ | no _ | no _
  -- -- ... | a | i = sel a i
  --   = sel a i , λ ρ j → pf ρ (eval e₁ ρ ++ j) ∙ cong (eval a ρ) (cong (_++ j) (q ρ))

  opt (E.imapb {s = s} {p = p} {q = q} x e) with opt e
  -- ... | (let′ {s = r} a b) , pf = (let′ (imap a) (E.imapb x
  --   (sub b ((skeep (sdrop sub-id)) ▹ sel (var v₁) (var v₀))))) , foo
  --   where
  --   foo : _
  --   foo ρ i =
  --     pf _ _
  --     ∙ sym (eval-sub b _ _ _
  --     ∙ eval-cong b
  --       (sub-env-wks (sdrop sub-id) (skip ⊆-eq) _ ▹ refl ▹ λ _ → refl) _
  --     ∙ eval-cong b (sub-env-sdrop sub-id ▹ refl ▹ λ _ → refl) _
  --     ∙ eval-cong b (sub-env-id ▹ refl ▹ λ _ → refl) _
  --     ∙ eval-cong b (wk-env-id ▹ refl ▹ λ _ → refl) _
  --     ∙ eval-cong b (reflᶜ ▹ refl ▹ foo') _)
  --     where
  --     foo' : _
  --     foo' l =
  --       cong (eval a _) (splitP-proj₂ {s} {j = l})
  --       ∙ cong (λ x → eval a (_ , x) _) (splitP-proj₁ {s} {j = l})

  ... | a , p = E.imapb x a , λ ρ j → p (ρ , ix-div j x) (ix-mod j x)


  opt (E.selb x e e₁) with opt e | opt e₁
  ... | a , p | i , q = E.selb x a i
                      , λ ρ j → p ρ (ix-combine (eval e₁ ρ) j x)
                                ∙ cong (eval a ρ) (cong (λ t → ix-combine t j x ) (q ρ))
  opt (E.sum {s = s}{p} e) with opt e
  -- Jairo made
  -- DANGER: maybe not efficient memorywise
  -- ... | a ⊞ b , pf = (E.sum {s = s}{p} a ⊞ E.sum {s = s}{p} b) , foo where
  --   foo : _
  --   foo ρ i =
  --     sum-inv {s = s} _+_ (fromℕ 0) i
  --     ∙ sum-cong{s = s} _+_ (fromℕ 𝟘) (λ _ → pf _ _)
  --     ∙ sum-dist {s = s} _+_ (fromℕ 𝟘) +-neutˡ +-medial
  --     ∙ cong₂ _+_
  --       (sym (sum-inv {s = s} _+_ (fromℕ 0) i))
  --       (sym (sum-inv {s = s} _+_ (fromℕ 0) i))

  -- ... | ⊟ a , pf = (⊟ (E.sum a))
  --   , λ ρ j →
  --   sum-inv _+_ (fromℕ 0) {λ i → eval e (ρ , i)} j
  --   ∙ sum-cong _+_ (fromℕ 0) (λ i → pf ((_ , i)) j)
  --   ∙ {!   !}
  ... | let′ {s = q} a b , pf = (let′ (imap a) (E.sum {s = s}
    (sub b (skeep (sdrop sub-id) ▹ sel (var v₁) (var v₀))))) , foo where
      foo : _
      foo ρ i = let aux = skeep (sdrop sub-id) ▹ sel (var v₁) (var v₀) in
        let aux1 = λ j → (((ρ , (λ i₁ → eval a (ρ , splitP i₁ .proj₁) (splitP i₁ .proj₂))) , j )) in
        sum-inv {s = s} _+_ (fromℕ 0) i
        ∙ sum-cong {s = s} _+_ (fromℕ 𝟘) (λ k → pf _ _)
        ∙ sym ( sum-inv {s = s} _+_ (fromℕ 0) i
        ∙ sum-cong {s = s} _+_ (fromℕ 𝟘) {λ j → eval (sub b _) (aux1 j) i}
          (λ k →
          (eval-sub b _ aux i)
          ∙ eval-cong b (sub-env-wks (sdrop sub-id) (skip ⊆-eq) _ ▹ refl ▹ λ _ → refl) i
          ∙ eval-cong b (sub-env-sdrop sub-id ▹ refl ▹ λ _ → refl) i
          ∙ eval-cong b (sub-env-id ▹ refl ▹ λ _ → refl) i
          ∙ eval-cong b (wk-env-id ▹ refl ▹ λ _ → refl) i
          ∙ eval-cong b (reflᶜ ▹ refl ▹ λ z →
            (cong (eval a _) (splitP-proj₂ {i = k}))
            ∙ cong (λ x → eval a (_ , x) _) (splitP-proj₁ {i = k})) i
          ))
  --... | zero = zero
  ... | zero , p = zero
                 , λ ρ j → sum-inv _+_ (fromℕ 0) {λ i → eval e (ρ , i)} j
                           ∙ sum-cong _+_ (fromℕ 0) {λ i → eval e (ρ , i) j} (λ i → p (ρ , i) j)
                           ∙ sum-zero {s}
  --... | imapₛ a = imapₛ (sum (ctx-swap v₁ a))
  ... | imaps a′ , pf = imaps (E.sum (sub a′ sub-swap))
                      , λ ρ j → let ss = (wks (wks sub-id (skip ⊆-eq))
                                              (skip (keep ⊆-eq)) ▹ var v₀) ▹ var v₁ in
                                sym (sum-inv _+_ (fromℕ 0)
                                             {(λ i → eval (sub a′ ss) ((ρ , j) , i))} []
                                     ∙ sum-cong _+_ (fromℕ 0)
                                                {(λ j₁ → eval (sub a′ ss) ((ρ , j) , j₁) [])}
                                                (λ k → eval-sub a′ ((ρ , j) , k) ss []
                                                       ∙ eval-cong a′ ((sub-env-wks _ _ ((ρ , j) , k)
                                                                        ∙ᶜ sub-env-wks _ _ (wk-env ⊆-eq ρ , j)
                                                                        ∙ᶜ sub-env-id ∙ᶜ wk-env-id ∙ᶜ wk-env-id) ▹ refl ▹ refl) [] )
                                     ∙ sum-cong _+_ (fromℕ 0) {λ z → eval a′ ((ρ , z) , j) []} (λ i → sym (pf (ρ , i) j))
                                     ∙ sym (sum-inv _+_ (fromℕ 0) {λ z → eval e (ρ , z)} j))
  --... | imap a = imap (sum (ctx-swap v₁ a))
  ... | imap a′ , pf = imap (E.sum (sub a′ sub-swap))
                     , λ ρ j → let ss = ((wks (wks sub-id (skip ⊆-eq))
                                              (skip (keep ⊆-eq)) ▹ var v₀)
                                         ▹ var v₁)
                               in sym (sum-inv _+_ (fromℕ 0) {λ i → eval (sub a′ ss) ((ρ , splitP j .proj₁) , i)} (splitP j .proj₂)
                                       ∙ sum-cong _+_ (fromℕ 0)
                                                  {λ j₁ → eval (sub a′ ss) ((ρ , splitP j .proj₁) , j₁) (splitP j .proj₂)}
                                                  (λ k → eval-sub a′ ((ρ , splitP j .proj₁) , k) ss (splitP j .proj₂)
                                                         ∙ eval-cong a′ ((sub-env-wks _ _ ((ρ , splitP j .proj₁) , k)
                                                                          ∙ᶜ sub-env-wks _ _ (wk-env ⊆-eq ρ , splitP j .proj₁)
                                                                          ∙ᶜ sub-env-id ∙ᶜ wk-env-id ∙ᶜ wk-env-id) ▹ refl ▹ refl)
                                                                        (splitP j .proj₂))
                                       ∙ sum-cong _+_ (fromℕ 0)  (λ i → sym (pf (ρ , i) j))
                                       ∙ sym (sum-inv _+_ (fromℕ 0) {λ z → eval e (ρ , z)} j))
  --... | imapb m a = imapb m (sum (ctx-swap v₁ a))
  ... | imapb m a′ , pf = E.imapb m (E.sum (sub a′ sub-swap))
                        , λ ρ j → let ss = ((wks (wks sub-id (skip ⊆-eq)) (skip (keep ⊆-eq)) ▹ var v₀) ▹ var v₁)
                                  in sym (sum-inv _+_ (fromℕ 0) {λ i → eval (sub a′ ss) ((ρ , ix-div j m) , i)} (ix-mod j m)
                                          ∙ sum-cong _+_ (fromℕ 0)
                                                     {λ j₁ → eval (sub a′ ss) ((ρ , ix-div j m) , j₁) (ix-mod j m)}
                                                     (λ k → eval-sub a′ ((ρ , ix-div j m) , k) ss (ix-mod j m)
                                                            ∙ eval-cong a′ ((sub-env-wks _ _ ((ρ , ix-div j m) , k)
                                                                            ∙ᶜ sub-env-wks _ _ (wk-env ⊆-eq ρ , ix-div j m)
                                                                            ∙ᶜ sub-env-id ∙ᶜ wk-env-id ∙ᶜ wk-env-id) ▹ refl ▹ refl)
                                                                           (ix-mod j m))
                                          ∙ sum-cong _+_ (fromℕ 0)  (λ i → sym (pf (ρ , i) j))
                                          ∙ sym (sum-inv _+_ (fromℕ 0) {λ z → eval e (ρ , z)} j))
  --... zero-but block ...
  ... | zero-but (var i) (var j) a , pf with eq? v₀ i | eq? v₀ j
  ... | veq  | veq = E.sum a , go
    where go : (ρ : ⟦ _ ⟧ᶜ) (i₁ : _)
             → Ar.sum _ _ (λ i₂ → eval e (ρ , i₂)) i₁
               ≡ Ar.sum _ _ (λ i₂ → eval a (ρ , i₂)) i₁
          go ρ j = sum-inv _+_ (fromℕ 0) {(λ i₂ → eval e (ρ , i₂))} j
                   ∙ sum-cong _+_ (fromℕ 0) {λ j₂ → eval e (ρ , j₂) j}
                                  (λ k → pf (ρ , k) j ∙ eval-zb a (var v₀) (ρ , k) j)
                   ∙ sym (sum-inv _+_ (fromℕ 0) {(λ z → eval a (ρ , z))} j)
  ... | neq _ i′ | veq = sub a (sub-id ▹ var i′)
                       , λ ρ j → sum-inv _+_ (fromℕ 0) {(λ i₁ → eval e (ρ , i₁))} j
                                 ∙ sum-cong _+_ (fromℕ 0) {λ j₂ → eval e (ρ , j₂) j}
                                                (λ k → pf (ρ , k) j ∙ zb-zbs (lookup i′ ρ) k j (λ k → eval a (ρ , k)))
                                 ∙ zbs-sum-s (lookup i′ ρ) _
                                 ∙ (sym (eval-sub a ρ (sub-id ▹ var i′) j
                                         ∙ eval-cong a (sub-env-id ▹ refl) j))
  ... | veq | neq _ j′ = sub a (sub-id ▹ var j′)
                       , λ ρ j → sum-inv _+_ (fromℕ 0) {(λ i₁ → eval e (ρ , i₁))} j
                                 ∙ sum-cong _+_ (fromℕ 0) {(λ j₂ → eval e (ρ , j₂) j)}
                                                (λ k → pf (ρ , k) j
                                                       ∙ zb-sym k _ j (λ k → eval a (ρ , k))
                                                       ∙ zb-zbs (lookup j′ ρ) k j (λ k → eval a (ρ , k)))
                                 ∙ zbs-sum-s (lookup j′ ρ) _
                                 ∙ (sym (eval-sub a ρ (sub-id ▹ var j′) j
                                         ∙ eval-cong a (sub-env-id ▹ refl) j))
  ... | neq _ i′ | neq _ j′ = zero-but (var i′) (var j′) (E.sum a)
                            , λ ρ j → sum-inv _+_ (fromℕ 0) {(λ i₁ → eval e (ρ , i₁))} j
                                      ∙ sum-cong _+_ (fromℕ 0) {(λ j₂ → eval e (ρ , j₂) j)}
                                                     (λ k → pf (ρ , k) j ∙ zb-zbs (lookup i′ ρ) _ j λ _ → eval a (ρ , k))
                                      ∙ zbs-ext (lookup i′ ρ) (lookup j′ ρ) (λ z → eval a (ρ , z) j)
                                      ∙ sym (zb-zbs-k (lookup i′ ρ) _ j  (Ar.sum (Ar.zipWith _+_) (K (fromℕ 0)) (λ i₁ → eval a (ρ , i₁)))
                                             ∙ zbs-cong _ _ (λ _ → sum-inv _+_ (fromℕ 0){(λ i₂ → eval a (ρ , i₂))} j) (lookup i′ ρ) (lookup j′ ρ))
  opt (E.sum {s = s} e) | a , p = E.sum a
              , λ ρ j → sum-inv _+_ (fromℕ 0) {(λ i₁ → eval e (ρ , i₁))} j
                        ∙ sum-cong _+_ (fromℕ 0) {λ j₁ → eval e (ρ , j₁) j} (λ i → p (ρ , i) j)
                        ∙ (sym (sum-inv _+_ (fromℕ 0) {λ i → eval a (ρ , i)} j))
  opt (zero-but e e₁ e₂) with opt e₂
  ... | a , p = zero-but e e₁ a
              , go
      where go : (ρ : ⟦ _ ⟧ᶜ) (j : _) → eval (zero-but e e₁ e₂) ρ j ≡ eval (zero-but e e₁ a) ρ j
            go ρ j with eval e ρ ≟ₚ eval e₁ ρ
            ... | yes _ = p ρ j
            ... | no _ = refl

  opt (E.slide e x e₁ x₁) with opt e₁
  -- TODO zero case
  ... | a , p = E.slide e x a x₁
              , λ ρ j → p ρ ((eval e ρ ⊕′ j) x₁ x)
  opt (E.backslide e e₁ x x₁) with opt e₁
  -- TODO zero case
  ... | a , p = E.backslide e a x x₁ , go
      where go : (ρ : ⟦ _ ⟧ᶜ) (j : _ ) →
                 (Ar.backslide (eval e ρ) (eval e₁ ρ) x (fromℕ 0) x₁ j)
                 ≡ eval (E.backslide e a x x₁) ρ j
            go ρ j with (j ⊝′ eval e ρ) x x₁
            ... | yes (k , _) = p ρ k
            ... | no _ = refl
  opt (logi e) with opt e
  -- TODO: imap(s) cases
  ... | a , p = logi a , λ ρ j → cong logisticʳ (p ρ j)
  opt (e ⊞ e₁) with opt e | opt e₁
  ... | a , p | b , q with isZero a
  ... | yes refl = b , λ ρ j → cong₂ _+_ (p ρ j) (q ρ j) ∙ +-neutˡ
  --... | a | zero = a
  ... | no _ with isZero b
  ... | yes refl = a , λ ρ j → cong₂ _+_ (p ρ j) (q ρ j) ∙ +-neutʳ

  ... | no _ with isImaps a
  ... | yes (a′ , refl) = imaps (a′ ⊞ sels (b ↑) (var here))
                        , λ ρ j → cong₂ _+_ (p ρ j)
                                            (sym (eval-wk (skip ⊆-eq) b (ρ , j) j
                                                  ∙ eval-cong b (wk-env-id) j
                                                  ∙ (sym (q ρ j))))
  --... | a , p | imaps b , q = imaps (sels (a ↑) (var here) ⊞ b)
  ... | no _ with isImaps b
  ... | yes (b′ , refl) = imaps (sels (a ↑) (var here) ⊞ b′)
                        , λ ρ j → cong₂ _+_ (p ρ j
                                             ∙ sym (eval-wk (skip ⊆-eq) a (ρ , j) j
                                                    ∙ eval-cong a (wk-env-id) j))
                                           (q ρ j)
  ... | no _ with (isImap a) | (isImap b)
  ... | yes (s , r , refl , a′ , refl) | _ = (imap (a′ ⊞ sel (b ↑) (var v₀)))
    , λ ρ j → cong₂ _+_ (p _ _) (sym ((eval-wk (skip ⊆-eq) b _ _)
    ∙ eval-cong b wk-env-id _
    ∙ cong (eval b _) (sym (splitP-eq {s = s} j))
    ∙ sym (q ρ j)
    ))
  ... | _ | yes (s , r , refl , b′ , refl)  = imap (sel (a ↑) (var v₀) ⊞ b′)
    , λ ρ j → cong₂ _+_
    ((p _ _)
    ∙ sym (eval-wk (skip ⊆-eq) a _ _
    ∙ eval-cong a (wk-env-id) _
    ∙ cong (eval a _) (sym (splitP-eq {s = s} j))
    ))
    (q _ _)
  -- ... | yes (i , j , x) | yes (i′ , j′ , x′) = {!   !}
  -- ... | _ | _ with (isImap a) | (isImap b)
  -- ... | yes c | yes d = {!   !}
  -- ... | zero-but (var i) (var j) x | zero-but (var i′) (var j′) x′ with eq? i i′ | eq? j j′
  -- ... | veq | veq = zero-but (var i) (var j) (x ⊞ x′)
  --                 , foo
  --     where foo : _
  --           foo ρ k rewrite p ρ k | q ρ k with lookup i ρ ≟ₚ lookup j ρ
  --           ... | yes _ = refl
  --           ... | no _ = +-neutʳ

  -- ... | _ | _ = zero-but (var i) (var j) x ⊞ zero-but (var i′) (var j′) x′ , λ ρ k → cong₂ _+_ (p ρ k) (q ρ k)

  opt (e ⊞ e₁) |
    (zero-but (var i) (var j) x) , p | (zero-but (var i′) (var j′) x′) , q
    | _ | _ | _ | _ | _ | _  with eq? i i′ | eq? j j′
  ... | veq | veq = zero-but (var i) (var j) (x ⊞ x′) , foo where
    foo : _
    foo ρ k rewrite p ρ k | q ρ k with lookup i ρ ≟ₚ lookup j ρ
    ... | yes _ = refl
    ... | no _ = +-neutʳ
  ... | _ | _ = zero-but (var i) (var j) x ⊞ zero-but (var i′) (var j′) x′ , λ ρ k → cong₂ _+_ (p ρ k) (q ρ k)

  opt (e ⊞ e₁) | a , p | b , q | _ | _ | _ | _ | _ | _ = a ⊞ b , λ ρ j → cong₂ _+_ (p ρ j) (q ρ j)
  -- Jairo Made
  opt (e ⊠ e₁) with opt e | opt e₁
  ... | one , p | b , q = b , λ ρ j → cong₂ _*_ (p ρ j) (q ρ j) ∙ *-neutˡ
  ... | (zero-but i j a) , p | b , q = zero-but i j (a ⊠ b) , foo
       where
       foo : _
       foo ρ k with eval i ρ ≟ₚ eval j ρ | (p ρ k) | (q ρ k)
       ... | yes _ | p′ | q′ rewrite p′ | q′ = refl
       ... | no _ | p′ | q′ rewrite p′ = *-nulˡ
  ... | a , p | b , q with isImaps b
  ... | yes (b′ , refl) = imaps (sels (a ↑) (var here) ⊠ b′)
                        , λ ρ j → cong₂ _*_ (p ρ j
                                             ∙ sym (eval-wk (skip ⊆-eq) a (ρ , j) j
                                                    ∙ eval-cong a (wk-env-id) j))
                                           (q ρ j)
  ... | no _ with isImaps a
  ... | yes (a′ , refl) = (imaps (a′ ⊠ sels (b ↑) (var here)))
                        , λ ρ j → cong₂ _*_ (p ρ j)
                                            (sym (eval-wk (skip ⊆-eq) b (ρ , j) j
                                                  ∙ eval-cong b (wk-env-id) j
                                                  ∙ (sym (q ρ j))))
  ... | no _ with isUn b
  ... | yes (inverse , (c ⊠ d) , refl) =
    maybe′
      (λ eq → 𝟙/ d , λ ρ i → cong₂ _*_ (p _ _ ∙ cong (λ x → eval x _ _) eq) (q _ _) ∙ *-÷-cut)
      (a ⊠ b , λ ρ j → cong₂ _*_ (p ρ j) (q ρ j)) (a ≟ᵉ c)
  ... | yes (u , c , refl) = a ⊠ b , λ ρ j → cong₂ _*_ (p ρ j) (q ρ j)
  ... | no _ = a ⊠ b , λ ρ j → cong₂ _*_ (p ρ j) (q ρ j)
  opt (scaledown x e) with opt e
  -- Jairo Made
  ... | (imaps {s = s} a) , q = imaps (scaledown x a)
      , λ ρ i → cong (_÷ fromℕ x) (q ρ i)
  ... | (imap {s = s} {p = p} a) , q = (imap {s = s} {p = p} (scaledown x a))
    , λ ρ i → cong (_÷ fromℕ x) (q ρ i)
  ... | (zero-but {s = s} {p = p} i j a) , q = foo where
    foo : _
    foo with (Data.Nat._≟_ x 0)
    ... | yes _ = (scaledown x (zero-but i j a)) , λ ρ j → cong (_÷ fromℕ x) (q ρ j)
    ... | no b = (zero-but i j (scaledown x a)) , foo' where
      foo' : _
      foo' ρ k with eval i ρ ≟ₚ eval j ρ | (cong (_÷ fromℕ x) (q ρ k))
      ... | yes b | q' = q'
      ... | no _ | q' rewrite q' = ÷-nul λ tt → b (fromℕ-inj tt)
  ... | a , p = scaledown x a , λ ρ j → cong (_÷ fromℕ x) (p ρ j)
  opt (⊟ e) with opt e
  ... | (zero-but i j a) , p = (zero-but i j (⊟ a)) , foo where
    foo : _
    foo ρ k with eval i ρ ≟ₚ eval j ρ | (cong -_ (p ρ k))
    ... | yes b | eq = eq
    ... | no b | eq = eq ∙ minus-idʳ
  ... | a , p = ⊟ a , λ ρ j → cong -_ (p ρ j)
  opt (let′ e e₁) with opt e | opt e₁
  ... | a , p | b , q with isVar a
  ... | yes (v , refl) = (sub b (sub-id ▹ var v))
    , λ ρ j → q _ j
      ∙ sym (eval-sub b ρ _ j
      ∙ eval-cong b (sub-env-id ▹ (λ i → sym $ p ρ i)) j)
  -- jairo made
  opt (let′ e e₁) | a , p | b , q | no _ with isLet a
  ... | yes (_ , c , d , refl) = (let′ c (let′ d (wk (keep (skip ⊆-eq)) b)))
      , foo where
      foo : _
      foo ρ i = (q (ρ , eval e ρ) i)
                ∙ sym ((eval-wk (keep (skip ⊆-eq)) b _ i)
                ∙ eval-cong b wk-env-id i
                ∙ eval-cong b (reflᶜ ▹ λ j → sym (p _ j)) i
                )

  ... | _ = let′ a b
          , λ ρ j → q (ρ , eval e ρ) j ∙ eval-cong b (reflᶜ ▹ p ρ) j
  opt (𝕖^ e) with opt e
  ... | a , p = 𝕖^ a , λ ρ i → cong e^_ (p ρ i)
  opt (relu e) with opt e
  ... | a , p = relu a , (λ ρ i → cong (_∨_ 0ᵣ) (p ρ i))
  opt (sqrt e) with opt e
  ... | a , p = sqrt a , λ ρ i → cong √_ (p ρ i)
  opt (𝟙/ e) with opt e
  ... | a , p = 𝟙/ a , (λ ρ i → cong 1/_ (p ρ i))
  opt (𝕀+ e) with opt e
  ... | a , p = 𝕀+ a , (λ ρ i → cong I+ (p ρ i))
  opt (ln e) with opt e
  ... | a , p = ln a , (λ ρ i → cong log (p ρ i))
  opt (argmax pf e) with opt e
  ... | a , p = (argmax pf a) , λ ρ → cong proj₂ (sum-cong max-pair (-∞ᵣ , lastIx pf) λ i → ×-≡,≡→≡ (p ρ i , refl))

  danger-opt : E Γ is → E Γ is
  danger-opt e =
    (opt e .proj₁)
    -- let-out $ sum-in $ (opt e .proj₁)
