-- {-# OPTIONS --warn=noUserWarning #-}
module _ where
  open import Ar hiding (sum; slide; backslide; imapb; selb)
  open import Relation.Binary.PropositionalEquality
  open import Data.Product
  open import Data.Nat using (ℕ; zero; suc; _≟_)
  open import Data.List as L
  open import Data.List.Properties as L
  open import Relation.Nullary
  open import Function
  open import Lang
  open import Ar
  -- open import Relation.Nullary.Irrelevant

  -- Equality of types
  _≟ⁱ_ : (a b : IS) → Dec (a ≡ b)
  ix x ≟ⁱ ix y with x ≟ˢ y
  ... | no ¬p = no λ { refl → ¬p refl }
  ... | yes refl = yes refl
  ix x ≟ⁱ ar y = no λ ()
  ar x ≟ⁱ ix y = no λ ()
  ar x ≟ⁱ ar y with x ≟ˢ y
  ... | no ¬p = no λ { refl → ¬p refl }
  ... | yes refl = yes refl

  _≟ᵏ_ : (a b : Kop) → Dec (a ≡ b)
  zero-op ≟ᵏ zero-op = yes refl
  zero-op ≟ᵏ one-op = no λ ()
  one-op ≟ᵏ zero-op = no λ ()
  one-op ≟ᵏ one-op = yes refl

  _≟ᵒ_ : (a b : Bop) → Dec (a ≡ b)
  plus-op ≟ᵒ plus-op = yes refl
  plus-op ≟ᵒ mul-op = no λ ()
  mul-op ≟ᵒ plus-op = no λ ()
  mul-op ≟ᵒ mul-op = yes refl

  _≟ᵘ_ : (a b : Uop) → Dec (a ≡ b)
  neg-op ≟ᵘ neg-op = yes refl
  neg-op ≟ᵘ relu-op = no λ ()
  neg-op ≟ᵘ sqrt-op = no λ ()
  neg-op ≟ᵘ inv-op = no λ ()
  neg-op ≟ᵘ ind-op = no λ ()
  neg-op ≟ᵘ ln-op = no λ ()
  neg-op ≟ᵘ softmax-op = no λ ()
  neg-op ≟ᵘ scaledown-op x = no λ ()
  relu-op ≟ᵘ neg-op = no λ ()
  relu-op ≟ᵘ relu-op = yes refl
  relu-op ≟ᵘ sqrt-op = no λ ()
  relu-op ≟ᵘ inv-op = no λ ()
  relu-op ≟ᵘ ind-op = no λ ()
  relu-op ≟ᵘ ln-op = no λ ()
  relu-op ≟ᵘ softmax-op = no λ ()
  relu-op ≟ᵘ scaledown-op x = no λ ()
  inv-op ≟ᵘ neg-op = no λ ()
  inv-op ≟ᵘ relu-op = no λ ()
  inv-op ≟ᵘ sqrt-op = no λ ()
  inv-op ≟ᵘ inv-op = yes refl
  inv-op ≟ᵘ ind-op = no λ ()
  inv-op ≟ᵘ ln-op = no λ ()
  inv-op ≟ᵘ softmax-op = no λ ()
  inv-op ≟ᵘ scaledown-op x = no λ ()
  sqrt-op ≟ᵘ neg-op = no λ ()
  sqrt-op ≟ᵘ relu-op = no λ ()
  sqrt-op ≟ᵘ sqrt-op = yes refl
  sqrt-op ≟ᵘ inv-op = no λ ()
  sqrt-op ≟ᵘ ind-op = no λ ()
  sqrt-op ≟ᵘ ln-op = no λ ()
  sqrt-op ≟ᵘ softmax-op = no λ ()
  sqrt-op ≟ᵘ scaledown-op x = no λ ()
  ind-op ≟ᵘ neg-op = no λ ()
  ind-op ≟ᵘ relu-op = no λ ()
  ind-op ≟ᵘ sqrt-op = no λ ()
  ind-op ≟ᵘ inv-op = no λ ()
  ind-op ≟ᵘ ind-op = yes refl
  ind-op ≟ᵘ ln-op = no λ ()
  ind-op ≟ᵘ softmax-op = no λ ()
  ind-op ≟ᵘ scaledown-op x = no λ ()
  ln-op ≟ᵘ neg-op = no λ ()
  ln-op ≟ᵘ relu-op = no λ ()
  ln-op ≟ᵘ sqrt-op = no λ ()
  ln-op ≟ᵘ inv-op = no λ ()
  ln-op ≟ᵘ ind-op = no λ ()
  ln-op ≟ᵘ ln-op = yes refl
  ln-op ≟ᵘ softmax-op = no λ ()
  ln-op ≟ᵘ scaledown-op x = no λ ()
  softmax-op ≟ᵘ neg-op = no λ ()
  softmax-op ≟ᵘ relu-op = no λ ()
  softmax-op ≟ᵘ sqrt-op = no λ ()
  softmax-op ≟ᵘ inv-op = no λ ()
  softmax-op ≟ᵘ ind-op = no λ ()
  softmax-op ≟ᵘ ln-op = no λ ()
  softmax-op ≟ᵘ softmax-op = yes refl
  softmax-op ≟ᵘ scaledown-op x = no λ ()
  scaledown-op x ≟ᵘ neg-op = no λ ()
  scaledown-op x ≟ᵘ relu-op = no λ ()
  scaledown-op x ≟ᵘ sqrt-op = no λ ()
  scaledown-op x ≟ᵘ inv-op = no λ ()
  scaledown-op x ≟ᵘ ind-op = no λ ()
  scaledown-op x ≟ᵘ ln-op = no λ ()
  scaledown-op x ≟ᵘ softmax-op = no λ ()
  scaledown-op x ≟ᵘ scaledown-op x₁ with x ≟ x₁
  ... | yes refl = yes refl
  ... | no a = no λ eq → a (scaledown-inj eq)

  -- Hail UIP
  *≈-uopiq : (a b : s * p ≈ q) → a ≡ b
  *≈-uopiq {[]} {[]} {[]} [] [] = refl
  *≈-uopiq {x ∷ s} {x₁ ∷ p} {x₂ ∷ q} (cons ⦃ refl ⦄ ⦃ a ⦄) (cons ⦃ refl ⦄ ⦃ b ⦄)
    = cong₂ (λ t u → cons ⦃ t ⦄ ⦃ u ⦄) refl (*≈-uopiq a b)

  +≈-uopiq : (a b : s + p ≈ q) → a ≡ b
  +≈-uopiq {[]} {[]} {[]} [] [] = refl
  +≈-uopiq {x ∷ s} {x₁ ∷ p} {x₂ ∷ q} (cons ⦃ refl ⦄ ⦃ a ⦄) (cons ⦃ refl ⦄ ⦃ b ⦄)
    = cong₂ (λ t u → cons ⦃ t ⦄ ⦃ u ⦄) refl (+≈-uopiq a b)

  suc≈-uopiq : (a b : suc s ≈ p) → a ≡ b
  suc≈-uopiq [] [] = refl
  suc≈-uopiq (cons ⦃ refl ⦄ ⦃ a ⦄) (cons ⦃ refl ⦄ ⦃ b ⦄) = cong₂ (λ t u → cons ⦃ t ⦄ ⦃ u ⦄) refl (suc≈-uopiq a b)

  uopiq : {a b : s ≡ p} → a ≡ b
  uopiq {a = refl} {b = refl} = refl

  _≟ᵐ_ : ∀ {s p q} (a : Mop s p q) → (b : Mop s p q) → Dec (a ≡ b)
  imaps-op ≟ᵐ imaps-op = yes refl
  imaps-op ≟ᵐ imap-op x = no λ ()
  imaps-op ≟ᵐ imapb-op x = no λ ()
  imaps-op ≟ᵐ sum-op = no λ ()
  imap-op x ≟ᵐ imaps-op = no λ ()
  imap-op x ≟ᵐ imap-op x₁ = yes (cong imap-op uopiq)
  imap-op x ≟ᵐ imapb-op x₁ = no λ ()
  imap-op x ≟ᵐ sum-op = no λ ()
  imapb-op x ≟ᵐ imaps-op = no λ ()
  imapb-op x ≟ᵐ imap-op x₁ = no λ ()
  imapb-op x ≟ᵐ imapb-op x₁ = yes (cong imapb-op (*≈-uopiq _ _))
  imapb-op x ≟ᵐ sum-op = no λ ()
  sum-op ≟ᵐ imaps-op = no λ ()
  sum-op ≟ᵐ imap-op x = no λ ()
  sum-op ≟ᵐ imapb-op x = no λ ()
  sum-op ≟ᵐ sum-op = yes refl

  _≟ᶜ_ : ∀ {s p q} (a : Sop s p q) → (b : Sop s p q) → Dec (a ≡ b)
  sels-op ≟ᶜ sels-op = yes refl
  sels-op ≟ᶜ sel-op x = no λ ()
  sels-op ≟ᶜ selb-op x = no λ ()
  sel-op x ≟ᶜ sels-op = no λ ()
  sel-op x ≟ᶜ sel-op x₁ = yes (cong sel-op uopiq)
  sel-op x ≟ᶜ selb-op x₁ = no λ ()
  selb-op x ≟ᶜ sels-op = no λ ()
  selb-op x ≟ᶜ sel-op x₁ = no λ ()
  selb-op x ≟ᶜ selb-op x₁ = yes (cong selb-op (*≈-uopiq _ _))

  isVar : (e : E Γ is) → Dec (∃ λ v → e ≡ var v)
  isVar (var x) = yes (x , refl)
  isVar 𝟘 = no λ ()
  isVar 𝟙 = no λ ()
  isVar (imaps e) = no λ ()
  isVar (sels e e₁) = no λ ()
  isVar (imap′ eq e) = no λ ()
  isVar (sel′ eq e e₁) = no λ ()
  isVar (imapb x e) = no λ ()
  isVar (selb x e e₁) = no λ ()
  isVar (Lang.sum e) = no λ ()
  isVar (zero-but e e₁ e₂) = no λ ()
  isVar (bop x e e₁) = no λ ()
  isVar (scaledown x e) = no λ ()
  isVar (let′ e e₁) = no λ ()
  isVar (uop x e) = no λ ()

  isKop : (e : E Γ (ar s)) → Dec (∃ λ t → e ≡ kop t)
  isKop (var x) = no λ ()
  isKop (kop x) = yes (x , refl)
  isKop (uop x e) = no λ ()
  isKop (bop x e e₁) = no λ ()
  isKop (mop x e) = no λ ()
  isKop (sop x e e₁) = no λ ()
  isKop (zero-but e e₁ e₂) = no λ ()
  isKop (let′ e e₁) = no λ ()

  isUop : (e : E Γ (ar s)) → Dec (∃₂ λ a b → e ≡ uop a b)
  isUop (var x) = no λ ()
  isUop (kop x) = no λ ()
  isUop (uop x e) = yes (x , e , refl)
  isUop (bop x e e₁) = no λ ()
  isUop (mop x e) = no λ ()
  isUop (sop x e e₁) = no λ ()
  isUop (zero-but e e₁ e₂) = no λ ()
  isUop (let′ e e₁) = no λ ()

  isBop : (e : E Γ (ar s)) → Dec (∃₂ λ o t → ∃ λ t₁ → e ≡ bop o t t₁)
  isBop (var x) = no λ ()
  isBop (kop x) = no λ ()
  isBop (uop x e) = no λ ()
  isBop (bop x e e₁) = yes (x , e , e₁ , refl)
  isBop (mop x e) = no λ ()
  isBop (sop x e e₁) = no λ ()
  isBop (zero-but e e₁ e₂) = no λ ()
  isBop (let′ e e₁) = no λ ()

  isMop : (e : E Γ (ar s))
    → Dec (∃₂ (λ p q → ∃₂ (λ x t → e ≡ mop {p} {q} x t)))
  isMop (var x) = no λ ()
  isMop (kop x) = no λ ()
  isMop (uop x e) = no λ ()
  isMop (bop x e e₁) = no λ ()
  isMop (mop {p} {q} x e) = yes (p , q , x , e , refl)
  isMop (sop x e e₁) = no λ ()
  isMop (zero-but e e₁ e₂) = no λ ()
  isMop (let′ e e₁) = no λ ()

  isSop : (e : E Γ (ar s))
    → Dec (∃₂ (λ p q → ∃ (λ x → ∃₂ (λ a b → e ≡ sop {p} {q} x a b))))
  isSop (var x) = no λ ()
  isSop (kop x) = no λ ()
  isSop (uop x e) = no λ ()
  isSop (bop x e e₁) = no λ ()
  isSop (mop x e) = no λ ()
  isSop (sop {p} {q} x e e₁) = yes (p , q , x , e , e₁ , refl)
  isSop (zero-but e e₁ e₂) = no λ ()
  isSop (let′ e e₁) = no λ ()

  isZeroBut : (e : E Γ (ar p)) → Dec (∃₂ λ s i → ∃₂ λ j u → e ≡ zero-but {s = s} i j u)
  isZeroBut (var x) = no λ ()
  isZeroBut (kop x) = no λ ()
  isZeroBut (uop x e) = no λ ()
  isZeroBut (bop x e e₁) = no λ ()
  isZeroBut (mop x e) = no λ ()
  isZeroBut (sop x e e₁) = no λ ()
  isZeroBut (zero-but {s = s} e e₁ e₂) = yes (s , e , e₁ , e₂ , refl)
  isZeroBut (let′ e e₁) = no λ ()

  isLet : (e : E Γ (ar p)) → Dec (∃₂ λ s′ t → ∃ λ t₁ → e ≡ let′ {s = s′} t t₁)
  isLet (var x) = no λ ()
  isLet (kop x) = no λ ()
  isLet (uop x e) = no λ ()
  isLet (bop x e e₁) = no λ ()
  isLet (mop x e) = no λ ()
  isLet (sop x e e₁) = no λ ()
  isLet (zero-but e e₁ e₂) = no λ ()
  isLet (let′ {s = s} e e₁) = yes (s , e , e₁ , refl)


  open import Data.Maybe

  _≟ᵉ_ : (a b : E Γ is) → Maybe (a ≡ b)
  var x ≟ᵉ u with isVar u
  ... | no ¬p = nothing
  ... | yes (v , refl) with eq? x v
  ... | neq _ _ = nothing
  ... | veq = just refl
  kop x ≟ᵉ b with isKop b
  ... | no _ = nothing
  ... | yes (y , refl) with x ≟ᵏ y
  ... | no _ = nothing
  ... | yes refl = just refl
  uop x a ≟ᵉ b with isUop b
  ... | no _ = nothing
  ... | yes (y , c , refl) with x ≟ᵘ y
  ... | no _ = nothing
  ... | yes refl = (a ≟ᵉ c) >>= just ∘ (cong (uop y))
  bop x a b ≟ᵉ c with isBop c
  ... | no _ = nothing
  ... | yes (y , d , e , refl) with x ≟ᵒ y
  ... | no _ = nothing
  ... | yes refl = do
    eq1 ← a ≟ᵉ d
    eq2 ← b ≟ᵉ e
    just (cong₂ (bop _) eq1 eq2)
  mop {s} {p} {q} x a ≟ᵉ b with isMop b
  ... | no _ = nothing
  ... | yes (s′ , p′ , y , a′ , refl) with s ≟ˢ s′ | p ≟ˢ p′
  ... | no _ | _ = nothing
  ... | _ | no _ = nothing
  ... | yes refl | yes refl with x ≟ᵐ y
  ... | no _ = nothing
  ... | yes refl = (a ≟ᵉ a′) >>= just ∘ (cong (mop _))
  sop {s} {p} {q} x a i ≟ᵉ b with isSop b
  ... | no _ = nothing
  ... | yes (s′ , p′ , y , a′ , i′ , refl) with s ≟ˢ s′ | p ≟ˢ p′
  ... | no _ | _ = nothing
  ... | _ | no _ = nothing
  ... | yes refl | yes refl with x ≟ᶜ y
  ... | no _ = nothing
  ... | yes refl = do
    c ← a ≟ᵉ a′
    d ← i ≟ᵉ i′
    just (cong₂ (sop _) c d)
  zero-but {s = s} e e₁ e₂ ≟ᵉ u with isZeroBut u
  ... | no ¬p = nothing
  ... | yes (s′ , i , j , u , refl) with s ≟ˢ s′
  ... | no ¬p = nothing
  ... | yes refl = do
    ei ← e ≟ᵉ i
    e₁j ← e₁ ≟ᵉ j
    e₂u ← e₂ ≟ᵉ u
    just (cong₃ zero-but ei e₁j e₂u)
  let′ {s = s} e e₁ ≟ᵉ u with isLet u
  ... | no ¬p = nothing
  ... | yes (s′ , t , t₁ , refl) with s ≟ˢ s′
  ... | no ¬p = nothing
  ... | yes refl = do
    et ← e ≟ᵉ t
    e₁t₁ ← e₁ ≟ᵉ t₁
    just (cong₂ let′ et e₁t₁)