-- {-# OPTIONS --warn=noUserWarning #-}
-- {-# OPTIONS --allow-unsolved-metas #-}
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

module XOpt (r : Real) (rp : RealProp r) where

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
  let-out (uop x e) = uop x (let-out e)
  let-out (bop x e e₁) = bop x (let-out e) (let-out e₁)
  let-out (mop x e) = let-out-step (mop x) (let-out e)
  let-out (sop x e e₁) = sop x (let-out e) (let-out e₁)
  let-out (zero-but e e₁ e₂) = zero-but (let-out e) (let-out e₁) (let-out e₂)
  let-out (let′ e e₁) = let′ (let-out e) (let-out e₁)
  let-out e = e

  sels-in : E Γ is → E Γ is
  sels-in (sels (⊟ e) e₁) = ⊟ (sels (sels-in e) (sels-in e₁))
  sels-in (sels (bop x a b) e₁) = bop x (sels (sels-in a) (sels-in e₁)) (sels (sels-in b) (sels-in e₁))
  sels-in (uop x e) = uop x (sels-in e)
  sels-in (bop x e e₁) = bop x (sels-in e) (sels-in e₁)
  sels-in (mop x e) = mop x (sels-in e)
  sels-in (sop x e e₁) = sop x (sels-in e) (sels-in e₁)
  sels-in (zero-but e e₁ e₂) = zero-but (sels-in e) (sels-in e₁) (sels-in e₂)
  sels-in (let′ e e₁) = let′ (sels-in e) (sels-in e₁)
  sels-in e = e

  opt : (e : E Γ is) → ∃ λ e′ → (e ≈ᵉ e′)
  opt (var v) = var v , reflᵉ (var v)
  opt (kop x) = kop x , reflᵉ (kop x)
  opt (⊟ e) with opt e
  ... | (⊟ a) , p =  a , (λ ρ i → (cong -_ (p _ _)) ∙ minus-invʳ)
  ... | (zero-but i j a) , p = (zero-but i j (⊟ a)) , foo where
    foo : _
    foo ρ k with eval i ρ ≟ₚ eval j ρ | (cong -_ (p ρ k))
    ... | yes b | eq = eq
    ... | no b | eq = eq ∙ minus-idʳ
  ... | a , p = ⊟ a , λ ρ j → cong -_ (p ρ j)
  opt (relu e) with opt e
  ... | a , p = relu a , (λ ρ i → cong (_∨_ 0ᵣ) (p ρ i))
  opt (√ e) with opt e
  ... | a , p = Lang.√ a , (λ ρ i → cong √_ (p ρ i))
  opt (𝟙/ e) with opt e
  ... | a , p = 𝟙/ a , (λ ρ i → cong 1/_ (p ρ i))
  opt (𝕚+ e) with opt e
  ... | a , p = 𝕚+ a , (λ ρ i → cong I+ (p ρ i))
  opt (ln e) with opt e
  ... | a , p = ln a , (λ ρ i → cong log (p ρ i))
  opt (ℙ e) with opt e
  ... | a , p = ℙ a , λ ρ i → cong₂ _÷_ (cong e^_ (p _ _))
    (sum-cong _+_ _ {e^_ ∘ (eval e ρ)} λ j → cong e^_ (p _ _))
  opt (scaledown x e) with opt e
  ... | (mop {s = s} {p = p} (imap-op refl) a) , q = (imap {s = s} {p = p} (scaledown x a))
    , λ ρ i → cong (_÷ fromℕ x) (q ρ i)
  ... | (zero-but {s = s} {p = p} i j a) , q = (zero-but i j (scaledown x a)) , foo where
    foo : _
    foo ρ k with eval i ρ ≟ₚ eval j ρ | (cong (_÷ fromℕ x) (q ρ k))
    ... | yes b | q' = q'
    ... | no _ | q' rewrite q' = ÷-nul
  ... | a , p = scaledown x a , λ ρ j → cong (_÷ fromℕ x) (p ρ j)
  opt (e ⊞ e₁) with opt e | opt e₁
  ... | 𝟘 , p | b , q = b , λ ρ j → cong₂ _+_ (p ρ j) (q ρ j) ∙ +-neutˡ
  ... | a , p | 𝟘 , q = a , λ ρ j → cong₂ _+_ (p ρ j) (q ρ j) ∙ +-neutʳ
  ... | (imaps a) , p | b , q = imaps (a ⊞ sels (b ↑) (var here))
                                , λ ρ j → cong₂ _+_ (p ρ j)
                                  (sym (eval-wk (skip ⊆-eq) b (ρ , j) j
                                        ∙ eval-cong b (wk-env-id) j
                                        ∙ (sym (q ρ j))))
  ... | a , p | (imaps b) , q = imaps (sels (a ↑) (var here) ⊞ b)
                                , λ ρ j → cong₂ _+_ (p ρ j
                                  ∙ sym (eval-wk (skip ⊆-eq) a (ρ , j) j
                                        ∙ eval-cong a (wk-env-id) j))
                                          (q ρ j)
  ... | (mop {s = s} {p = r} (imap-op refl) a) , p | b , q =
    (imap (a ⊞ sel (b ↑) (var v₀))) , λ ρ j → cong₂ _+_ (p _ _) (sym ((eval-wk (skip ⊆-eq) b _ _)
    ∙ eval-cong b wk-env-id _
    ∙ cong (eval b _) (sym (splitP-eq {s = s} j))
    ∙ sym (q ρ j)
    ))
  ... | a , p | (mop {s = s} {p = r} (imap-op refl) b) , q = imap (sel (a ↑) (var v₀) ⊞ b)
    , λ ρ j → cong₂ _+_
    ((p _ _)
    ∙ sym (eval-wk (skip ⊆-eq) a _ _
    ∙ eval-cong a (wk-env-id) _
    ∙ cong (eval a _) (sym (splitP-eq {s = s} j))
    ))
    (q _ _)
  ... | (zero-but (var i) (var j) a) , p
      | (zero-but (var i′) (var j′) b) , q = foo where
      foo : _
      foo with eq? i i′ | eq? j j′
      ... | veq | veq = zero-but (var i) (var j) (a ⊞ b) , foo' where
        foo' : _
        foo' ρ k rewrite p ρ k | q ρ k with lookup i ρ ≟ₚ lookup j ρ
        ... | yes _ = refl
        ... | no _ = +-neutʳ
      ... | _ | _ = zero-but (var i) (var j) a ⊞ zero-but (var i′) (var j′) b , λ ρ k → cong₂ _+_ (p ρ k) (q ρ k)
  ... | a , p | b , q = a ⊞ b , λ ρ j → cong₂ _+_ (p ρ j) (q ρ j)
  opt (e ⊠ e₁) with opt e | opt e₁
  ... | 𝟙 , p | b , q = b , λ ρ j → cong₂ _*_ (p ρ j) (q ρ j) ∙ *-neutˡ
  ... | (zero-but i j a) , p | b , q = (zero-but i j (a ⊠ b)) , foo where
    foo : _
    foo ρ k with eval i ρ ≟ₚ eval j ρ | (p ρ k) | (q ρ k)
    ... | yes _ | p′ | q′ rewrite p′ | q′ = refl
    ... | no _ | p′ | q′ rewrite p′ = *-nulˡ
  ... | a , p | (imaps b) , q =
    imaps (sels (a ↑) (var here) ⊠ b)
    , λ ρ j → cong₂ _*_ (p ρ j
      ∙ sym (eval-wk (skip ⊆-eq) a (ρ , j) j
      ∙ eval-cong a (wk-env-id) j))
      (q ρ j)
  ... | (imaps a) , p | b , q =
    (imaps (a ⊠ sels (b ↑) (var here)))
    , λ ρ j → cong₂ _*_ (p ρ j)
      (sym (eval-wk (skip ⊆-eq) b (ρ , j) j
      ∙ eval-cong b (wk-env-id) j
      ∙ (sym (q ρ j))))
  ... | a , p | (⊟ b) , q = (⊟ (a ⊠ b)) , λ ρ i → (cong₂ _*_ (p ρ i) (q ρ i)) ∙ minus-*-pushʳ
  ... | a , p | (zero-but i j b) , q = zero-but i j (a ⊠ b) , foo
    where
    foo : _
    foo ρ k with eval i ρ ≟ₚ eval j ρ | (p ρ k) | (q ρ k)
    ... | yes _ | p′ | q′ rewrite p′ | q′ = refl
    ... | no _ | p′ | q′ rewrite q′ = *-nulʳ
  ... | a , p | b , q = a ⊠ b , λ ρ j → cong₂ _*_ (p ρ j) (q ρ j)
  opt (imaps e) with opt e
  ... | a , p = imaps a , λ ρ i → p (ρ , i) []
  opt (imap′ {s = s} {p = p} refl e) with opt e
  ... | (let′ a b) , pf = foo where
    foo : _
    foo with (stren-∃ a v₀)
    ... | nothing = imap (let′ a b) , λ ρ i → pf (ρ , splitP i .proj₁) (splitP i .proj₂)
    ... | just (c , eq) = (let′ c (imap (sub b sub-swap))) , foo' where
      foo' : _
      foo' ρ i = let j , k = splitP {s} i .proj₁ , splitP {s} i .proj₂ in
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
  ... | t , pf = imap t , λ ρ i → pf (ρ , splitP i .proj₁) (splitP i .proj₂)
  opt (imapb {s = s} {p = p} {q = q} x e) with opt e
  ... | a , p = Lang.imapb x a , λ ρ j → p (ρ , ix-div j x) (ix-mod j x)
  opt (sum {s = s}{p} e) with opt e
  ... | let′ {s = q} a b , pf = (let′ (imap a) (Lang.sum {s = s}
    (sub b (skeep (sdrop sub-id) ▹ sel (var v₁) (var v₀))))) , foo where
      foo : _
      foo ρ i = let aux = skeep (sdrop sub-id) ▹ sel (var v₁) (var v₀) in
        let aux1 = λ j → (((ρ , (λ i₁ → eval a (ρ , splitP i₁ .proj₁) (splitP i₁ .proj₂))) , j )) in
        sum-inv {s = s} _+_ (fromℕ 0) i
        ∙ sum-cong {s = s} _+_ (fromℕ 0) (λ k → pf _ _)
        ∙ sym ( sum-inv {s = s} _+_ (fromℕ 0) i
        ∙ sum-cong {s = s} _+_ (fromℕ 0) {λ j → eval (sub b _) (aux1 j) i}
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
  ... | 𝟘 , p = 𝟘
                 , λ ρ j → sum-inv _+_ (fromℕ 0) {λ i → eval e (ρ , i)} j
                           ∙ sum-cong _+_ (fromℕ 0) {λ i → eval e (ρ , i) j} (λ i → p (ρ , i) j)
                           ∙ sum-zero {s}
  ... | imaps a′ , pf = imaps (Lang.sum (sub a′ sub-swap))
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
  ... | imapb m a′ , pf = Lang.imapb m (Lang.sum (sub a′ sub-swap))
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
  ... | zero-but (var i) (var j) a , pf = foo where
    a′ = zero-but (var i) (var j) a
    foo : _
    foo with eq? v₀ i | eq? v₀ j
    ... | veq  | veq = Lang.sum a , go where
      go : _
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
    ... | neq _ i′ | neq _ j′ = zero-but (var i′) (var j′) (Lang.sum a)
                            , λ ρ j → sum-inv _+_ (fromℕ 0) {(λ i₁ → eval e (ρ , i₁))} j
                                      ∙ sum-cong _+_ (fromℕ 0) {(λ j₂ → eval e (ρ , j₂) j)}
                                                     (λ k → pf (ρ , k) j ∙ zb-zbs (lookup i′ ρ) _ j λ _ → eval a (ρ , k))
                                      ∙ zbs-ext (lookup i′ ρ) (lookup j′ ρ) (λ z → eval a (ρ , z) j)
                                      ∙ sym (zb-zbs-k (lookup i′ ρ) _ j  (Ar.sum (Ar.zipWith _+_) (K (fromℕ 0)) (λ i₁ → eval a (ρ , i₁)))
                                             ∙ zbs-cong _ _ (λ _ → sum-inv _+_ (fromℕ 0){(λ i₂ → eval a (ρ , i₂))} j) (lookup i′ ρ) (lookup j′ ρ))
  ... | a , p = Lang.sum a
              , λ ρ j → sum-inv _+_ (fromℕ 0) {(λ i₁ → eval e (ρ , i₁))} j
                        ∙ sum-cong _+_ (fromℕ 0) {λ j₁ → eval e (ρ , j₁) j} (λ i → p (ρ , i) j)
                        ∙ (sym (sum-inv _+_ (fromℕ 0) {λ i → eval a (ρ , i)} j))
  opt (sels e e₁) with opt e | opt e₁
  ... | (𝟘 , p)        | (i , q) = 𝟘 , λ ρ i → p ρ _
  ... | (𝟙 , p)         | (i , q) = 𝟙 , λ ρ i → p ρ _
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
  ... | (sum e , p) | (i , q) = Lang.sum (sels e (i ↑))
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
  ... | zero-but {Γ = Γ} i j a , p | (k , q) = zero-but i j (sels a k)
                                     , go
          where
            go : (ρ : ⟦ Γ ⟧ᶜ) → ∀ u → eval e ρ (eval e₁ ρ) ≡ eval (zero-but i j (sels a k)) ρ u
            go ρ u with eval i ρ ≟ₚ eval j ρ | p ρ (eval e₁ ρ)
            ... | yes _ | p′ = p′ ∙ cong (eval a ρ) (q ρ)
            ... | no _ | p′ = p′
  ... | (⊟ e₂ , p) | (i , q) = (⊟ (sels e₂ i)) , λ ρ j → (p _ _) ∙ cong (λ x → - (eval e₂ ρ x)) (q _)
  ... | (𝟙/ e₂ , p) | (i , q) = (𝟙/ (sels e₂ i)) , λ ρ j → (p _ _) ∙ cong (λ x → fromℕ 1 ÷  (eval e₂ ρ x)) (q _)
  ... | a , p | i , q = sels a i , λ ρ j → p ρ (eval e₁ ρ) ∙ cong (eval a ρ) (q ρ)
  opt (sel′ {s = s} {p = p} {q = .(s ⊗ p)} refl e e₁) with opt e | opt e₁
  ... | 𝟘 , pf | i , qf = 𝟘 , λ ρ j → pf ρ (eval e₁ ρ ++ j)
  ... | 𝟙 , pf | i , qf = 𝟙 , λ ρ j → pf ρ (eval e₁ ρ ++ j)
  ... | (zero-but j k e₂) , pf | i , qf = zero-but j k (sel e₂ i) , go
       where
       go : _
       go ρ u with eval j ρ ≟ₚ eval k ρ | pf ρ (eval e₁ ρ ++ u)
       ... | yes _ | p′ = p′ ∙ cong (λ x → eval e₂ ρ (x ++ u)) (qf ρ)
       ... | no _ | p′ = p′
  ... | (let′ {s = r} c d) , pf | i , qf = let′ c (sel d (i ↑)) , foo where
        foo : _
        foo ρ j =
          pf ρ _
          ∙ sym (cong (λ x → eval d _ (x ++ j)) (eval-wk (skip ⊆-eq) i _
          ∙ eval-cong i wk-env-id
          ∙ sym (qf _)))
  ... | (imap′ {s = s′} {p = p′} {q = .(s ⊗ p)} eq u) , pf | i , qf = foo where
    a = imap′ eq u
    foo : _
    foo with s ≟ˢ s′
    ... | no _ = sel a i , λ ρ j → pf ρ (eval e₁ ρ ++ j) ∙ cong (eval a ρ) (cong (_++ j) (qf ρ))
    ... | yes refl with (++-inj₂ {s = s} eq)
    ... | refl rewrite eq = sub u (sub-id ▹ i) , go
        where go : (ρ : ⟦ _ ⟧ᶜ) (j : P p′) → eval e ρ (eval e₁ ρ ++ j) ≡ eval (sub u (sub-id ▹ _)) ρ _
              go ρ j rewrite qf ρ  = pf ρ (eval i ρ ++ j)
                                     ∙ sym (eval-sub u ρ (sub-id ▹ i) j
                                            ∙ eval-cong u (sub-env-id ▹ (sym $ splitP-proj₁ {j = j})) j
                                            ∙ cong (eval u _) (sym $ splitP-proj₂ {i = eval i ρ}))
  ... | a , pf | i , qf = sel a i , λ ρ j → pf ρ (eval e₁ ρ ++ j) ∙ cong (eval a ρ) (cong (_++ j) (qf ρ))
  opt (selb x e e₁) with opt e | opt e₁
  ... | a , p | i , q = Lang.selb x a i
                      , λ ρ j → p ρ (ix-combine (eval e₁ ρ) j x)
                                ∙ cong (eval a ρ) (cong (λ t → ix-combine t j x ) (q ρ))
  opt (zero-but i j e₂) with opt e₂ | i ≟ᵉ j
  ... | a , p | just refl =
    a , go where
      go : _
      go ρ k = (eval-zb e₂ i _ _) ∙ p _ _
  ... | a , p | nothing = zero-but i j a , go
      where go : (ρ : ⟦ _ ⟧ᶜ) (k : _) → eval (zero-but i j e₂) ρ k ≡ eval (zero-but i j a) ρ k
            go ρ k with eval i ρ ≟ₚ eval j ρ
            ... | yes _ = p ρ k
            ... | no _ = refl
  opt (let′ e e₁) with opt e | opt e₁
  ... | (let′ c d) , p | b , q = (let′ c (let′ d (wk (keep (skip ⊆-eq)) b)))
      , foo where
      foo : _
      foo ρ i = (q (ρ , eval e ρ) i)
                ∙ sym ((eval-wk (keep (skip ⊆-eq)) b _ i)
                ∙ eval-cong b wk-env-id i
                ∙ eval-cong b (reflᶜ ▹ λ j → sym (p _ j)) i
                )
  ... | a , p | b , q = let′ a b
          , λ ρ j → q (ρ , eval e ρ) j ∙ eval-cong b (reflᶜ ▹ p ρ) j

  danger-opt : E Γ is → E Γ is
  danger-opt e =
    -- (opt e .proj₁)
    sels-in $ let-out $ (opt e .proj₁)