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

  _≟ᵒ_ : (a b : Bop) → Dec (a ≡ b)
  plus ≟ᵒ plus = yes refl
  plus ≟ᵒ mul = no λ ()
  mul ≟ᵒ plus = no λ ()
  mul ≟ᵒ mul = yes refl

  _≟ᵘ_ : (a b : Uop) → Dec (a ≡ b)
  logistic ≟ᵘ logistic = yes refl
  logistic ≟ᵘ neg = no λ ()
  logistic ≟ᵘ exp = no λ ()
  logistic ≟ᵘ rectifier = no λ ()
  logistic ≟ᵘ squared = no λ ()
  logistic ≟ᵘ inverse = no λ ()
  logistic ≟ᵘ ind-positive = no λ ()
  logistic ≟ᵘ logarithm = no λ ()
  neg ≟ᵘ logistic = no λ ()
  neg ≟ᵘ neg = yes refl
  neg ≟ᵘ exp = no λ ()
  neg ≟ᵘ rectifier = no λ ()
  neg ≟ᵘ squared = no λ ()
  neg ≟ᵘ inverse = no λ ()
  neg ≟ᵘ ind-positive = no λ ()
  neg ≟ᵘ logarithm = no λ ()
  exp ≟ᵘ logistic = no λ ()
  exp ≟ᵘ neg = no λ ()
  exp ≟ᵘ exp = yes refl
  exp ≟ᵘ rectifier = no λ ()
  exp ≟ᵘ squared = no λ ()
  exp ≟ᵘ inverse = no λ ()
  exp ≟ᵘ ind-positive = no λ ()
  exp ≟ᵘ logarithm = no λ ()
  rectifier ≟ᵘ logistic = no λ ()
  rectifier ≟ᵘ neg = no λ ()
  rectifier ≟ᵘ exp = no λ ()
  rectifier ≟ᵘ rectifier = yes refl
  rectifier ≟ᵘ squared = no λ ()
  rectifier ≟ᵘ inverse = no λ ()
  rectifier ≟ᵘ ind-positive = no λ ()
  rectifier ≟ᵘ logarithm = no λ ()
  squared ≟ᵘ logistic = no λ ()
  squared ≟ᵘ neg = no λ ()
  squared ≟ᵘ exp = no λ ()
  squared ≟ᵘ rectifier = no λ ()
  squared ≟ᵘ squared = yes refl
  squared ≟ᵘ inverse = no λ ()
  squared ≟ᵘ ind-positive = no λ ()
  squared ≟ᵘ logarithm = no λ ()
  inverse ≟ᵘ logistic = no λ ()
  inverse ≟ᵘ neg = no λ ()
  inverse ≟ᵘ exp = no λ ()
  inverse ≟ᵘ rectifier = no λ ()
  inverse ≟ᵘ squared = no λ ()
  inverse ≟ᵘ inverse = yes refl
  inverse ≟ᵘ ind-positive = no λ ()
  inverse ≟ᵘ logarithm = no λ ()
  ind-positive ≟ᵘ logistic = no λ ()
  ind-positive ≟ᵘ neg = no λ ()
  ind-positive ≟ᵘ exp = no λ ()
  ind-positive ≟ᵘ rectifier = no λ ()
  ind-positive ≟ᵘ squared = no λ ()
  ind-positive ≟ᵘ inverse = no λ ()
  ind-positive ≟ᵘ ind-positive = yes refl
  ind-positive ≟ᵘ logarithm = no λ ()
  logarithm ≟ᵘ logistic = no λ ()
  logarithm ≟ᵘ neg = no λ ()
  logarithm ≟ᵘ exp = no λ ()
  logarithm ≟ᵘ rectifier = no λ ()
  logarithm ≟ᵘ squared = no λ ()
  logarithm ≟ᵘ inverse = no λ ()
  logarithm ≟ᵘ ind-positive = no λ ()
  logarithm ≟ᵘ logarithm = yes refl

  isVar : (e : E Γ is) → Dec (∃ λ v → e ≡ var v)
  isVar (var x) = yes (x , refl)
  isVar zero = no λ ()
  isVar one = no λ ()
  isVar (imaps e) = no λ ()
  isVar (sels e e₁) = no λ ()
  isVar (imap e) = no λ ()
  isVar (sel e e₁) = no λ ()
  isVar (imapb x e) = no λ ()
  isVar (selb x e e₁) = no λ ()
  isVar (E.sum e) = no λ ()
  isVar (zero-but e e₁ e₂) = no λ ()
  isVar (slide e x e₁ x₁) = no λ ()
  isVar (backslide e e₁ x x₁) = no λ ()
  isVar (bin x e e₁) = no λ ()
  isVar (scaledown x e) = no λ ()
  isVar (let′ e e₁) = no λ ()
  -- Jairo made
  isVar (un x e) = no λ ()
  isVar (maximum e) = no λ ()

  isZero : (e : E Γ (ar s)) → Dec (e ≡ zero)
  isZero zero = yes refl
  isZero (var x) = no  λ ()
  isZero one = no λ ()
  isZero (imaps e) = no λ ()
  isZero (sels e e₁) = no λ ()
  isZero (imap e) = no λ ()
  isZero (sel e e₁) = no λ ()
  isZero (E.imapb x e) = no λ ()
  isZero (E.selb x e e₁) = no λ ()
  isZero (E.sum e) = no λ ()
  isZero (zero-but e e₁ e₂) = no λ ()
  isZero (E.slide e x e₁ x₁) = no λ ()
  isZero (E.backslide e e₁ x x₁) = no λ ()
  isZero (bin x e e₁) = no λ ()
  isZero (scaledown x e) = no λ ()
  isZero (let′ e e₁) = no λ ()
  -- Jairo made
  isZero (un x e) = no λ ()
  isZero (maximum e) = no λ ()

  isOne : (e : E Γ (ar s)) → Dec (e ≡ one)
  isOne zero = no λ ()
  isOne (var x) = no λ ()
  isOne one = yes refl
  isOne (imaps e) = no λ ()
  isOne (sels e e₁) = no λ ()
  isOne (imap e) = no λ ()
  isOne (sel e e₁) = no λ ()
  isOne (E.imapb x e) = no λ ()
  isOne (E.selb x e e₁) = no λ ()
  isOne (E.sum e) = no λ ()
  isOne (zero-but e e₁ e₂) = no λ ()
  isOne (E.slide e x e₁ x₁) = no λ ()
  isOne (E.backslide e e₁ x x₁) = no λ ()
  isOne (bin x e e₁) = no λ ()
  isOne (scaledown x e) = no λ ()
  isOne (let′ e e₁) = no λ ()
  -- Jairo made
  isOne (un x e) = no λ ()
  isOne (maximum e) = no λ ()

  isImap : (e : E Γ (ar q))
         → Dec (∃₂ λ s p
                → Σ (s L.++ p ≡ q) λ eq → ∃ λ u → subst (E Γ ∘ ar) (sym eq) e ≡ imap {s = s} u)
  isImap (var x) = no λ { (_ , _ , refl , _ , ()) }
  isImap zero = no λ { (_ , _ , refl , _ , ()) }
  isImap one = no λ { (_ , _ , refl , _ , ()) }
  isImap (imaps e) = no λ { (_ , _ , refl , _ , ()) }
  isImap (sels e e₁) = no λ { ([] , [] , refl , _ , ())  }
  isImap (imap e) = yes (_ , _ , refl , e , refl)
  isImap (sel e e₁) = no λ { (_ , _ , refl , _ , ()) }
  isImap (E.imapb x e) = no λ { (_ , _ , refl , _ , ()) }
  isImap (E.selb x e e₁) = no λ { (_ , _ , refl , _ , ()) }
  isImap (E.sum e) = no λ { (_ , _ , refl , _ , ()) }
  isImap (zero-but e e₁ e₂) = no λ { (_ , _ , refl , _ , ()) }
  isImap (E.slide e x e₁ x₁) = no λ { (_ , _ , refl , _ , ()) }
  isImap (E.backslide e e₁ x x₁) = no λ { (_ , _ , refl , _ , ()) }
  isImap (bin x e e₁) = no λ { (_ , _ , refl , _ , ()) }
  isImap (scaledown x e) = no λ { (_ , _ , refl , _ , ()) }
  isImap (let′ e e₁) = no λ { (_ , _ , refl , _ , ()) }
  isImap (un x e) = no λ { (_ , _ , refl , _ , ()) }
  isImap (maximum e) = no λ { (_ , _ , refl , _ , ()) }


  isImaps : (e : E Γ (ar s)) → Dec (∃ λ u → e ≡ imaps u)
  isImaps (var x) = no λ ()
  isImaps zero = no λ ()
  isImaps one = no λ ()
  isImaps (imaps e) = yes (e , refl)
  isImaps (sels e e₁) = no λ ()
  isImaps (imap e) = no λ ()
  isImaps (sel e e₁) = no λ ()
  isImaps (E.imapb x e) = no λ ()
  isImaps (E.selb x e e₁) = no λ ()
  isImaps (E.sum e) = no λ ()
  isImaps (zero-but e e₁ e₂) = no λ ()
  isImaps (E.slide e x e₁ x₁) = no λ ()
  isImaps (E.backslide e e₁ x x₁) = no λ ()
  isImaps (bin x e e₁) = no λ ()
  isImaps (scaledown x e) = no λ ()
  isImaps (let′ e e₁) = no λ ()
  isImaps (un x e) = no λ ()
  isImaps (maximum e) = no λ ()

  isZeroBut : (e : E Γ (ar p)) → Dec (∃₂ λ s i → ∃₂ λ j u → e ≡ zero-but {s = s} i j u)
  isZeroBut (var x) = no λ ()
  isZeroBut zero = no λ ()
  isZeroBut one = no λ ()
  isZeroBut (imaps e) = no λ ()
  isZeroBut (sels e e₁) = no λ ()
  isZeroBut (imap e) = no λ ()
  isZeroBut (sel e e₁) = no λ ()
  isZeroBut (E.imapb x e) = no λ ()
  isZeroBut (E.selb x e e₁) = no λ ()
  isZeroBut (E.sum e) = no λ ()
  isZeroBut (zero-but e e₁ e₂) = yes (_ , e , e₁ , e₂ , refl)
  isZeroBut (E.slide e x e₁ x₁) = no λ ()
  isZeroBut (E.backslide e e₁ x x₁) = no λ ()
  isZeroBut (bin x e e₁) = no λ ()
  isZeroBut (scaledown x e) = no λ ()
  isZeroBut (let′ e e₁) = no λ ()
  isZeroBut (un x e) = no λ ()
  isZeroBut (maximum e) = no λ ()

  isSels : (e : E Γ (ar p)) (s : S) → Dec (Σ (p ≡ []) λ eq → ∃₂ λ t u → subst (E Γ ∘ ar) eq e ≡ sels {s = s} t u)
  isSels (var x) s = no λ { (refl , _ , _ , ()) }
  isSels zero s = no λ { (refl , _ , _ , ()) }
  isSels one s = no λ { (refl , _ , _ , ()) }
  isSels (imaps e) s = no λ { (refl , _ , _ , ()) }
  isSels (sels {s = s′} e e₁) s with s′ ≟ˢ s
  ... | no ¬p = no λ { (refl , t , u , refl) → ¬p refl }
  ... | yes refl = yes (refl , e , e₁ , refl)
  isSels (imap {s = s} e) s′ = no foo
    where foo : _
          foo (eq , t , u , x) with ++-[]₁ {s = s} eq
          foo (eq , t , u , x) | rr rewrite rr | eq with x
          ... | ()
  isSels (sel e e₁) s = no λ { (refl , _ , _ , ()) }
  isSels (E.imapb x e) s = no λ { (refl , _ , _ , ()) }
  isSels (E.selb x e e₁) s = no λ { (refl , _ , _ , ()) }
  isSels (E.sum e) s = no λ { (refl , _ , _ , ()) }
  isSels (zero-but e e₁ e₂) s = no λ { (refl , _ , _ , ()) }
  isSels (E.slide e x e₁ x₁) s = no λ { (refl , _ , _ , ()) }
  isSels (E.backslide e e₁ x x₁) s = no λ { (refl , _ , _ , ()) }
  isSels (bin x e e₁) s = no λ { (refl , _ , _ , ()) }
  isSels (scaledown x e) s = no λ { (refl , _ , _ , ()) }
  isSels (let′ e e₁) s = no λ { (refl , _ , _ , ()) }
  isSels (un x e) s = no λ { (refl , _ , _ , ()) }
  isSels (maximum e) s = no λ { (refl , _ , _ , ()) }

  isSel :  (e : E Γ (ar p)) → Dec (∃ λ s → ∃₂ λ t u → e ≡ sel {s = s}{p} t u)
  isSel (var x) = no λ { (_ , _ , _ , ()) }
  isSel zero = no λ { (_ , _ , _ , ()) }
  isSel one = no λ { (_ , _ , _ , ()) }
  isSel (imaps e) = no λ { (_ , _ , _ , ()) }
  isSel (sels e e₁) = no λ { (_ , _ , _ , ()) }
  isSel (imap e) = no λ { (_ , _ , _ , ()) }
  isSel (sel e e₁) = yes (_ , e , e₁ , refl)
  isSel (E.imapb x e) = no λ { (_ , _ , _ , ()) }
  isSel (E.selb x e e₁) = no λ { (_ , _ , _ , ()) }
  isSel (E.sum e) = no λ { (_ , _ , _ , ()) }
  isSel (zero-but e e₁ e₂) = no λ { (_ , _ , _ , ()) }
  isSel (E.slide e x e₁ x₁) = no λ { (_ , _ , _ , ()) }
  isSel (E.backslide e e₁ x x₁) = no λ { (_ , _ , _ , ()) }
  isSel (bin x e e₁) = no λ { (_ , _ , _ , ()) }
  isSel (scaledown x e) = no λ { (_ , _ , _ , ()) }
  isSel (let′ e e₁) = no λ { (_ , _ , _ , ()) }
  isSel (un x e) = no λ { (_ , _ , _ , ()) }
  isSel (maximum e) = no λ { (_ , _ , _ , ()) }

  isImapb : (e : E Γ (ar q)) → Dec (∃₂ λ s p → Σ (s * p ≈ q) λ pf → ∃ λ t → e ≡ E.imapb pf t)
  isImapb (var x) = no λ { (_ , _ , _ , _ , ()) }
  isImapb zero = no λ { (_ , _ , _ , _ , ()) }
  isImapb one = no λ { (_ , _ , _ , _ , ()) }
  isImapb (imaps e) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (sels e e₁) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (imap e) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (sel e e₁) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (E.imapb x e) = yes (_ , _ , x , e , refl)
  isImapb (E.selb x e e₁) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (E.sum e) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (zero-but e e₁ e₂) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (E.slide e x e₁ x₁) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (E.backslide e e₁ x x₁) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (bin x e e₁) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (scaledown x e) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (let′ e e₁) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (un x e) = no λ { (_ , _ , _ , _ , ()) }
  isImapb (maximum e) = no λ { (_ , _ , _ , _ , ()) }

  isSelb : (e : E Γ (ar p)) → Dec (∃₂ λ s q → Σ (s * p ≈ q) λ pf → ∃₂ λ t u → e ≡ E.selb pf t u)
  isSelb (var x) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb zero = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb one = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (imaps e) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (sels e e₁) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (imap e) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (sel e e₁) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (E.imapb x e) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (E.selb x e e₁) = yes (_ , _ , x , e , e₁ , refl)
  isSelb (E.sum e) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (zero-but e e₁ e₂) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (E.slide e x e₁ x₁) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (E.backslide e e₁ x x₁) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (bin x e e₁) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (scaledown x e) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (let′ e e₁) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (un x e) = no λ { (_ , _ , _ , _ , _ , ()) }
  isSelb (maximum e) = no λ { (_ , _ , _ , _ , _ , ()) }

  isSum : (e : E Γ (ar p)) → Dec (∃₂ λ s t → e ≡ E.sum {s = s} t)
  isSum (var x) = no λ ()
  isSum zero = no λ ()
  isSum one = no λ ()
  isSum (imaps e) = no λ ()
  isSum (sels e e₁) = no λ ()
  isSum (imap e) = no λ ()
  isSum (sel e e₁) = no λ ()
  isSum (E.imapb x e) = no λ ()
  isSum (E.selb x e e₁) = no λ ()
  isSum (E.sum e) = yes (_ , e , refl)
  isSum (zero-but e e₁ e₂) = no λ ()
  isSum (E.slide e x e₁ x₁) = no λ ()
  isSum (E.backslide e e₁ x x₁) = no λ ()
  isSum (bin x e e₁) = no λ ()
  isSum (scaledown x e) = no λ ()
  isSum (let′ e e₁) = no λ ()
  isSum (un x e) = no λ ()
  isSum (maximum e) = no λ ()

  isMaximum : (e : E Γ (ar p)) → Dec (∃₂ λ s t → e ≡ maximum {s = s} t)
  isMaximum (var x) = no λ ()
  isMaximum zero = no λ ()
  isMaximum one = no λ ()
  isMaximum (imaps e) = no λ ()
  isMaximum (sels e e₁) = no λ ()
  isMaximum (imap e) = no λ ()
  isMaximum (sel e e₁) = no λ ()
  isMaximum (E.imapb x e) = no λ ()
  isMaximum (E.selb x e e₁) = no λ ()
  isMaximum (E.sum e) = no λ ()
  isMaximum (zero-but e e₁ e₂) = no λ ()
  isMaximum (E.slide e x e₁ x₁) = no λ ()
  isMaximum (E.backslide e e₁ x x₁) = no λ ()
  isMaximum (bin x e e₁) = no λ ()
  isMaximum (scaledown x e) = no λ ()
  isMaximum (let′ e e₁) = no λ ()
  isMaximum (un x e) = no λ ()
  isMaximum (maximum e) = yes (_ , e , refl)

  isSlide : (e : E Γ (ar u)) → Dec (∃₂ λ s′ p′ → ∃₂ λ r′ t → ∃₂ λ x′ t₁ → ∃ λ x₁ → e ≡ E.slide {s = s′}{p′}{r′} t x′ t₁ x₁)
  isSlide (var x) = no λ ()
  isSlide zero = no λ ()
  isSlide one = no λ ()
  isSlide (imaps e) = no λ ()
  isSlide (sels e e₁) = no λ ()
  isSlide (imap e) = no λ ()
  isSlide (sel e e₁) = no λ ()
  isSlide (E.imapb x e) = no λ ()
  isSlide (E.selb x e e₁) = no λ ()
  isSlide (E.sum e) = no λ ()
  isSlide (zero-but e e₁ e₂) = no λ ()
  isSlide (E.slide e x e₁ x₁) =  yes (_ , _ , _ , e , x , e₁ , x₁ , refl)
  isSlide (E.backslide e e₁ x x₁) = no λ ()
  isSlide (logi e) = no λ ()
  isSlide (bin x e e₁) = no λ ()
  isSlide (scaledown x e) = no λ ()
  isSlide (⊟ e) = no λ ()
  isSlide (let′ e e₁) = no λ ()
  isSlide (un x e) = no λ ()
  isSlide (maximum e) = no λ ()

  isBackslide : (e : E Γ (ar r))
              → Dec (∃₂ λ s′ u′ → ∃₂ λ p′ t → ∃₂ λ t₁ x → ∃ λ x₁
                     → e ≡ E.backslide {s = s′}{u = u′}{p = p′} t t₁ x x₁)
  isBackslide (var x) = no λ ()
  isBackslide zero = no λ ()
  isBackslide one = no λ ()
  isBackslide (imaps e) = no λ ()
  isBackslide (sels e e₁) = no λ ()
  isBackslide (imap e) = no λ ()
  isBackslide (sel e e₁) = no λ ()
  isBackslide (E.imapb x e) = no λ ()
  isBackslide (E.selb x e e₁) = no λ ()
  isBackslide (E.sum e) = no λ ()
  isBackslide (zero-but e e₁ e₂) = no λ ()
  isBackslide (E.slide e x e₁ x₁) = no λ ()
  isBackslide (E.backslide e e₁ x x₁) = yes (_ , _ , _ , e , e₁ , x , x₁ , refl)
  isBackslide (logi e) = no λ ()
  isBackslide (bin x e e₁) = no λ ()
  isBackslide (scaledown x e) = no λ ()
  isBackslide (⊟ e) = no λ ()
  isBackslide (let′ e e₁) = no λ ()
  isBackslide (un x e) = no λ ()
  isBackslide (maximum e) = no λ ()

  isUn : (e : E Γ (ar s)) → Dec (∃ λ t → ∃ λ t₁ → e ≡ un t t₁)
  isUn (var x) = no λ ()
  isUn 𝟘 = no λ ()
  isUn 𝟙 = no λ ()
  isUn (imaps e) = no λ ()
  isUn (sels e e₁) = no λ ()
  isUn (imap e) = no λ ()
  isUn (sel e e₁) = no λ ()
  isUn (E.imapb x e) = no λ ()
  isUn (E.selb x e e₁) = no λ ()
  isUn (E.sum e) = no λ ()
  isUn (zero-but e e₁ e₂) = no λ ()
  isUn (E.slide e x e₁ x₁) = no λ ()
  isUn (E.backslide e e₁ x x₁) = no λ ()
  isUn (bin x e e₁) = no λ ()
  isUn (scaledown x e) = no λ ()
  isUn (let′ e e₁) = no λ ()
  isUn (un x e) = yes (x , e , refl)
  isUn (maximum e) = no λ ()

  isBin : (e : E Γ (ar s)) → Dec (∃₂ λ o t → ∃ λ t₁ → e ≡ bin o t t₁)
  isBin (var x) = no λ ()
  isBin zero = no λ ()
  isBin one = no λ ()
  isBin (imaps e) = no λ ()
  isBin (sels e e₁) = no λ ()
  isBin (imap e) = no λ ()
  isBin (sel e e₁) = no λ ()
  isBin (E.imapb x e) = no λ ()
  isBin (E.selb x e e₁) = no λ ()
  isBin (E.sum e) = no λ ()
  isBin (zero-but e e₁ e₂) = no λ ()
  isBin (E.slide e x e₁ x₁) = no λ ()
  isBin (E.backslide e e₁ x x₁) = no λ ()
  isBin (logi e) = no λ ()
  isBin (bin x e e₁) = yes (x , e , e₁ , refl)
  isBin (scaledown x e) = no λ ()
  isBin (⊟ e) = no λ ()
  isBin (let′ e e₁) = no λ ()
  isBin (un x e) = no λ ()
  isBin (maximum e) = no λ ()

  isScaledown : (e : E Γ (ar s)) → Dec (∃₂ λ x t  → e ≡ scaledown x t)
  isScaledown (var x) = no λ ()
  isScaledown zero = no λ ()
  isScaledown one = no λ ()
  isScaledown (imaps e) = no λ ()
  isScaledown (sels e e₁) = no λ ()
  isScaledown (imap e) = no λ ()
  isScaledown (sel e e₁) = no λ ()
  isScaledown (E.imapb x e) = no λ ()
  isScaledown (E.selb x e e₁) = no λ ()
  isScaledown (E.sum e) = no λ ()
  isScaledown (zero-but e e₁ e₂) = no λ ()
  isScaledown (E.slide e x e₁ x₁) = no λ ()
  isScaledown (E.backslide e e₁ x x₁) = no λ ()
  isScaledown (logi e) = no λ ()
  isScaledown (bin x e e₁) = no λ ()
  isScaledown (scaledown x e) = yes (x , e , refl)
  isScaledown (⊟ e) = no λ ()
  isScaledown (let′ e e₁) = no λ ()
  isScaledown (un x e) = no λ ()
  isScaledown (maximum e) = no λ ()

  {-
  isMinus : (e : E Γ (ar s)) → Dec (∃ λ t  → e ≡ ⊟ t)
  isMinus (var x) = no λ ()
  isMinus zero = no λ ()
  isMinus one = no λ ()
  isMinus (imaps e) = no λ ()
  isMinus (sels e e₁) = no λ ()
  isMinus (imap e) = no λ ()
  isMinus (sel e e₁) = no λ ()
  isMinus (E.imapb x e) = no λ ()
  isMinus (E.selb x e e₁) = no λ ()
  isMinus (E.sum e) = no λ ()
  isMinus (zero-but e e₁ e₂) = no λ ()
  isMinus (E.slide e x e₁ x₁) = no λ ()
  isMinus (E.backslide e e₁ x x₁) = no λ ()
  isMinus (logi e) = no λ ()
  isMinus (bin x e e₁) = no λ ()
  isMinus (scaledown x e) = no λ ()
  isMinus (⊟ e) = yes (e , refl)
  isMinus (let′ e e₁) = no λ ()
  isMinus = {!   !}
  -}

  isLet : (e : E Γ (ar p)) → Dec (∃₂ λ s′ t → ∃ λ t₁ → e ≡ let′ {s = s′} t t₁)
  isLet (var x) = no λ ()
  isLet zero = no λ ()
  isLet one = no λ ()
  isLet (imaps e) = no λ ()
  isLet (sels e e₁) = no λ ()
  isLet (imap e) = no λ ()
  isLet (sel e e₁) = no λ ()
  isLet (E.imapb x e) = no λ ()
  isLet (E.selb x e e₁) = no λ ()
  isLet (E.sum e) = no λ ()
  isLet (zero-but e e₁ e₂) = no λ ()
  isLet (E.slide e x e₁ x₁) = no λ ()
  isLet (E.backslide e e₁ x x₁) = no λ ()
  isLet (logi e) = no λ ()
  isLet (bin x e e₁) = no λ ()
  isLet (scaledown x e) = no λ ()
  isLet (⊟ e) = no λ ()
  isLet (let′ e e₁) = yes (_ , e , e₁ , refl)
  isLet (un x e) = no λ ()
  isLet (maximum e) = no λ ()

  unvar : {x y : is ∈ Γ} → var x ≡ var y → x ≡ y
  unvar refl = refl

  -- Hail UIP
  *≈-uniq : (a b : s * p ≈ q) → a ≡ b
  *≈-uniq {[]} {[]} {[]} [] [] = refl
  *≈-uniq {x ∷ s} {x₁ ∷ p} {x₂ ∷ q} (cons ⦃ refl ⦄ ⦃ a ⦄) (cons ⦃ refl ⦄ ⦃ b ⦄)
    = cong₂ (λ t u → cons ⦃ t ⦄ ⦃ u ⦄) refl (*≈-uniq a b)

  +≈-uniq : (a b : s + p ≈ q) → a ≡ b
  +≈-uniq {[]} {[]} {[]} [] [] = refl
  +≈-uniq {x ∷ s} {x₁ ∷ p} {x₂ ∷ q} (cons ⦃ refl ⦄ ⦃ a ⦄) (cons ⦃ refl ⦄ ⦃ b ⦄)
    = cong₂ (λ t u → cons ⦃ t ⦄ ⦃ u ⦄) refl (+≈-uniq a b)

  suc≈-uniq : (a b : suc s ≈ p) → a ≡ b
  suc≈-uniq [] [] = refl
  suc≈-uniq (cons ⦃ refl ⦄ ⦃ a ⦄) (cons ⦃ refl ⦄ ⦃ b ⦄) = cong₂ (λ t u → cons ⦃ t ⦄ ⦃ u ⦄) refl (suc≈-uniq a b)

  open import Data.Maybe

  _≟ᵉ_ : (a b : E Γ is) → Maybe (a ≡ b)
  var x ≟ᵉ u with isVar u
  ... | no ¬p = nothing
  ... | yes (v , refl) with eq? x v
  ... | neq _ _ = nothing
  ... | veq = just refl
  zero ≟ᵉ u with isZero u
  ... | no ¬p = nothing
  ... | yes refl = just refl
  one ≟ᵉ u with isOne u
  ... | no ¬p = nothing
  ... | yes refl = just refl
  imaps e ≟ᵉ u with isImaps u
  ... | no ¬p = nothing
  ... | yes (u′ , refl) = e ≟ᵉ u′ >>= just ∘ (cong imaps)
  sels {s = s} e e₁ ≟ᵉ u with isSels u s
  ... | no ¬p = nothing
  ... | yes (refl , u , u₁ , refl) = do
    eu ← e ≟ᵉ u
    e₁u₁ ← e₁ ≟ᵉ u₁
    just (cong₂ sels eu e₁u₁)
  imap {s = s}{p} e ≟ᵉ u with isImap u
  ... | no ¬p = nothing
  ... | yes (s′ , p′ , spq , u , eq) with s ≟ˢ s′
  ... | no ¬p = nothing
  ... | yes refl with ++-cancelˡ s′ p′ p spq
  (imap {_} {s′} {p} e ≟ᵉ u₁) | yes (s′ , p , refl , u , refl) | yes refl | refl = e ≟ᵉ u >>= just ∘ (cong imap)
  sel {s = s} e e₁ ≟ᵉ u with isSel u
  ... | no ¬p = nothing
  ... | yes (s′ , u , u₁ , refl) with s ≟ˢ s′
  ... | no ¬p = nothing
  ... | yes refl = do
    eu ← e ≟ᵉ u
    e₁u₁ ← e₁ ≟ᵉ u₁
    just (cong₂ sel eu e₁u₁)
  E.imapb {s = s}{p} x e ≟ᵉ u with isImapb u
  ... | no ¬p = nothing
  ... | yes (s′ , p′ , x′ , t , refl) with s ≟ˢ s′
  ... | no ¬p = nothing
  ... | yes refl with p ≟ˢ p′
  ... | no ¬p = nothing
  ... | yes refl rewrite *≈-uniq x x′ = e ≟ᵉ t >>= just ∘ (cong (E.imapb x′))
  E.selb {s = s}{q = q} x e e₁ ≟ᵉ u with isSelb u
  ... | no ¬p = nothing
  ... | yes (s′ , q′ , x′ , u , u₁ , refl) with s ≟ˢ s′
  ... | no ¬p = nothing
  ... | yes refl with q ≟ˢ q′
  ... | no ¬p = nothing
  ... | yes refl rewrite *≈-uniq x x′ = do
    eu ← e ≟ᵉ u
    e₁u₁ ← e₁ ≟ᵉ u₁
    just (cong₂ (E.selb x′) eu e₁u₁)
  E.sum {s = s} e ≟ᵉ u with isSum u
  ... | no ¬p = nothing
  ... | yes (s′ , u , refl) with s ≟ˢ s′
  ... | no ¬p = nothing
  ... | yes refl = e ≟ᵉ u >>= just ∘ (cong E.sum)
  zero-but {s = s} e e₁ e₂ ≟ᵉ u with isZeroBut u
  ... | no ¬p = nothing
  ... | yes (s′ , i , j , u , refl) with s ≟ˢ s′
  ... | no ¬p = nothing
  ... | yes refl = do
    ei ← e ≟ᵉ i
    e₁j ← e₁ ≟ᵉ j
    e₂u ← e₂ ≟ᵉ u
    just (cong₃ zero-but ei e₁j e₂u)
  E.slide {s = s}{p}{r} e x e₁ x₁ ≟ᵉ w with isSlide w
  ... | no ¬p = nothing
  ... | yes (s′ , p′ , r′ , t , x′ , t₁ , x₁′ , refl) with s ≟ˢ s′ | p ≟ˢ p′ | r ≟ˢ r′
  ... | yes refl | yes refl | yes refl rewrite +≈-uniq x x′ | suc≈-uniq x₁ x₁′ = do
    et ← e ≟ᵉ t
    e₁t₁ ← e₁ ≟ᵉ t₁
    just (cong₂ (λ a b → E.slide a _ b _) et e₁t₁)
  ... | _ | _ | _ = nothing
  E.backslide {s = s}{u}{p} e e₁ x x₁ ≟ᵉ w with isBackslide w
  ... | no ¬p = nothing
  ... | yes (s′ , u′ , p′ , t , t₁ , x′ , x₁′ , refl) with s ≟ˢ s′ | u ≟ˢ u′ | p ≟ˢ p′
  ... | yes refl | yes refl | yes refl rewrite suc≈-uniq x x′ | +≈-uniq x₁ x₁′ = do
    et ← e ≟ᵉ t
    e₁t₁ ← e₁ ≟ᵉ t₁
    just (cong₂ (λ a b → E.backslide a b _ _ ) et e₁t₁)
  ... | _ | _ | _ = nothing
  -- logi e ≟ᵉ u with isLogistic u
  -- ... | no ¬p = nothing
  -- ... | yes (t , refl) = e ≟ᵉ t >>= just ∘ (cong logi)
  bin x e e₁ ≟ᵉ u with isBin u
  ... | no ¬p = nothing
  ... | yes (o , t , t₁ , refl) with x ≟ᵒ o
  ... | no ¬p = nothing
  ... | yes refl = do
    et ← e ≟ᵉ t
    e₁t₁ ← e₁ ≟ᵉ t₁
    just (cong₂ (bin _) et e₁t₁)
  scaledown x e ≟ᵉ u with isScaledown u
  ... | no ¬p = nothing
  ... | yes (x′ , t , refl) with x ≟ x′
  ... | no ¬p = nothing
  ... | yes refl = e ≟ᵉ t >>= just ∘ (cong (scaledown _))
  -- (⊟ e) ≟ᵉ u with isMinus u
  -- ... | no ¬p = nothing
  -- ... | yes (t , refl) = e ≟ᵉ t >>= just ∘ (cong (⊟_))
  let′ {s = s} e e₁ ≟ᵉ u with isLet u
  ... | no ¬p = nothing
  ... | yes (s′ , t , t₁ , refl) with s ≟ˢ s′
  ... | no ¬p = nothing
  ... | yes refl = do
    et ← e ≟ᵉ t
    e₁t₁ ← e₁ ≟ᵉ t₁
    just (cong₂ let′ et e₁t₁)
  -- Jairo made
  un x e ≟ᵉ u with isUn u
  ... | no ¬p = nothing
  ... | yes (o , t , refl) with x ≟ᵘ o
  ... | no ¬p = nothing
  ... | yes refl with e ≟ᵉ t
  ... | just refl = just (cong (un _) refl)
  ... | nothing = nothing
    -- do
    -- et ← e ≟ᵉ t -- ???
    -- just (cong (un o) et)
  maximum {s = s} e ≟ᵉ u with isMaximum u
  ... | no ¬p = nothing
  ... | yes (s′ , u , refl) with s ≟ˢ s′
  ... | no ¬p = nothing
  ... | yes refl = e ≟ᵉ u >>= just ∘ (cong maximum)

  e-eq? : (a : E Γ is) (b : E Γ ip) → Maybe (Σ (is ≡ ip) λ pp → subst (E Γ) pp a ≡ b)
  e-eq? {is = is}{ip} a b with is ≟ⁱ ip
  ... | no ¬p = nothing
  ... | yes refl = a ≟ᵉ b >>= just ∘ (refl ,_)


  open WkSub hiding (_∙ˢ_)
  -- stren? : ∀ {Γ is} (x : is ∈ Γ) (y : E Γ is) → Dec (∃ (λ y' → stren y x ≡ just y'))
  -- stren? x y with (stren y x)
  -- ... | just a = yes (a , refl)
  -- ... | nothing = no λ ()

  open import Data.Unit
  open import Data.Empty

  IsE₂ : ∀ {Γ' Γ is ip} (x : E Γ' is) (e : E Γ ip) → Set
  IsE₂ {_} {Γ} {_} {_} (var x) e = (∃ λ v → e ≡ var v)
  IsE₂ {_} {Γ} {ar s} {ix q} x e = ⊥
  IsE₂ {_} {Γ} {ar s} {ar q} 𝟘 e = (e ≡ zero)
  IsE₂ {_} {Γ} {ar s} {ar q} 𝟙 e = (e ≡ one)
  IsE₂ {_} {Γ} {ar s} {ar q} (imaps x) e = (∃ λ u → e ≡ imaps u)
  IsE₂ {_} {Γ} {ar s} {ar p} (sels x x₁) e = (Σ (p ≡ []) λ eq → ∃₂ λ t u → subst (E Γ ∘ ar) eq e ≡ sels {s = s} t u)
  IsE₂ {_} {Γ} {ar s} {ar q} (imap x) e = (∃₂ λ s p → Σ (s L.++ p ≡ q) λ eq → ∃ λ u → subst (E Γ ∘ ar) (sym eq) e ≡ imap {s = s} u)
  IsE₂ {_} {Γ} {ar s} {ar p} (sel x x₁) e =  (∃ λ s → ∃₂ λ t u → e ≡ sel {s = s}{p} t u)
  IsE₂ {_} {Γ} {ar s} {ar q} (E.imapb x x₁) e = (∃₂ λ s p → Σ (s * p ≈ q) λ pf → ∃ λ t → e ≡ E.imapb pf t)
  IsE₂ {_} {Γ} {ar s} {ar p} (E.selb x x₁ x₂) e = (∃₂ λ s q → Σ (s * p ≈ q) λ pf → ∃₂ λ t u → e ≡ E.selb pf t u)
  IsE₂ {_} {Γ} {ar s} {ar q} (E.sum x) e = (∃₂ λ s t → e ≡ E.sum {s = s} t)
  IsE₂ {_} {Γ} {ar s} {ar q} (zero-but x x₁ x₂) e = (∃₂ λ s i → ∃₂ λ j u → e ≡ zero-but {s = s} i j u)
  IsE₂ {_} {Γ} {ar s} {ar q} (E.slide x x₁ x₂ x₃) e = (∃₂ λ s′ p′ → ∃₂ λ r′ t → ∃₂ λ x′ t₁ → ∃ λ x₁ → e ≡ E.slide {s = s′}{p′}{r′} t x′ t₁ x₁)
  IsE₂ {_} {Γ} {ar s} {ar q} (E.backslide x x₁ x₂ x₃) e = (∃₂ λ s′ u′ → ∃₂ λ p′ t → ∃₂ λ t₁ x → ∃ λ x₁ → e ≡ E.backslide {s = s′}{u = u′}{p = p′} t t₁ x x₁)
  IsE₂ {_} {Γ} {ar s} {ar q} (bin x x₁ x₂) e = (∃₂ λ o t → ∃ λ t₁ → e ≡ bin o t t₁)
  IsE₂ {_} {Γ} {ar s} {ar q} (scaledown x x₁) e = (∃₂ λ x t  → e ≡ scaledown x t)
  IsE₂ {_} {Γ} {ar s} {ar q} (let′ x x₁) e = (∃₂ λ s′ t → ∃ λ t₁ → e ≡ let′ {s = s′} t t₁)
  IsE₂ {_} {Γ} {ar s} {ar q} (un x x₁) e = (∃ λ t → ∃ λ t₁ → e ≡ un t t₁)
  IsE₂ {_} {Γ} {ar s} {ar q} (maximum x) e = (∃₂ λ s t → e ≡ maximum {s = s} t)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (var x) e = (∃ λ v → e ≡ var v)

  isE₂ : ∀ {Γ' Γ is} (x : E Γ' is) (e : E Γ is) → Dec (IsE₂ x e)
  isE₂ {_} {_} {ar s} 𝟘 e = isZero e
  isE₂ {_} {_} {ar s} 𝟙 e = isOne e
  isE₂ {_} {_} {ar s} (imaps x) e = isImaps e
  isE₂ {_} {_} {ar s} (sels x x₁) e = isSels e s
  isE₂ {_} {_} {ar s} (imap x) e = isImap e
  isE₂ {_} {_} {ar s} (sel x x₁) e = isSel e
  isE₂ {_} {_} {ar s} (E.imapb x x₁) e = isImapb e
  isE₂ {_} {_} {ar s} (E.selb x x₁ x₂) e = isSelb e
  isE₂ {_} {_} {ar s} (E.sum x) e = isSum e
  isE₂ {_} {_} {ar s} (zero-but x x₁ x₂) e = isZeroBut e
  isE₂ {_} {_} {ar s} (E.slide x x₁ x₂ x₃) e = isSlide e
  isE₂ {_} {_} {ar s} (E.backslide x x₁ x₂ x₃) e = isBackslide e
  isE₂ {_} {_} {ar s} (bin x x₁ x₂) e = isBin e
  isE₂ {_} {_} {ar s} (scaledown x x₁) e = isScaledown e
  isE₂ {_} {_} {ar s} (let′ x x₁) e = isLet e
  isE₂ {_} {_} {ar s} (un x x₁) e = isUn e
  isE₂ {_} {_} {ar s} (maximum x) e = isMaximum e
  isE₂ {_} {_} {ix s} (var x) e = isVar e
  isE₂ {_} {_} {ar s} (var x) e = isVar e

  open import Data.Maybe renaming (map to mmap)
  open import Data.Maybe.Properties

  -- strenv? : (x : is ∈ Γ) (y : ip ∈ Γ)
  --   → Dec (∃ (λ (z : ip ∈ (Γ / x)) → y ≡ wkv (wk-/ x) z))
  -- strenv? v₀ v₀ = no λ ()
  -- strenv? v₀ (there y) = yes (y , cong there (sym (wkv-at-eq y)))
  -- strenv? (there x) v₀ = yes (v₀ , refl)
  -- strenv? (there x) (there y) =
  --   map′ f g (strenv? x y) where
  --     f : _
  --     f (a , b) = there a , cong there b

  --     g : _
  --     g (there a , b) = _ , v-inj b

  -- stren-∃ : (e : E Γ is) (v : ip ∈ Γ)
  --   → Maybe (∃ λ (z : E (Γ / v) is) → e ≡ wk (wk-/ v) z)
  -- stren-∃ (var x) v =
  --   mmap (λ (a , b) → _ , (cong var b)) (dec⇒maybe (strenv? v x))
  -- stren-∃ zero v = just (zero , refl)
  -- stren-∃ one v = just (one , refl)
  -- stren-∃ (imaps e) v =
  --   mmap (λ (a , b) → _ , (cong imaps b)) (stren-∃ e (there v))
  -- stren-∃ (sels e e₁) v = do
  --   (a , b) ← stren-∃ e v
  --   (c , d) ← stren-∃ e₁ v
  --   just (_ , (cong₂ sels b d))
  -- stren-∃ (imap e) v =
  --   mmap (λ (a , b) → _ , (cong imap b)) (stren-∃ e (there v))
  -- stren-∃ (sel e e₁) v = do
  --   (a , b) ← stren-∃ e v
  --   (c , d) ← stren-∃ e₁ v
  --   just (_ , (cong₂ sel b d))
  -- stren-∃ (imapb x e) v =
  --   mmap (λ (a , b) → _ , (cong (E.imapb x) b)) (stren-∃ e (there v))
  -- stren-∃ (selb x e e₁) v = do
  --   (a , b) ← stren-∃ e v
  --   (c , d) ← stren-∃ e₁ v
  --   just (_ , (cong₂ (E.selb x) b d))
  -- stren-∃ (sum e) v =
  --   mmap (λ (a , b) → _ , (cong E.sum b)) (stren-∃ e (there v))
  -- stren-∃ (zero-but e e₁ e₂) v = do
  --   (a , b) ← stren-∃ e v
  --   (c , d) ← stren-∃ e₁ v
  --   (e , f) ← stren-∃ e₂ v
  --   just (_ , cong₃ zero-but b d f)
  -- stren-∃ (slide e x e₁ x₁) v = do
  --   (a , b) ← stren-∃ e v
  --   (c , d) ← stren-∃ e₁ v
  --   just (_ , (cong₂ (λ f g → E.slide f x g x₁) b d))
  -- stren-∃ (backslide e e₁ x x₁) v = do
  --   (a , b) ← stren-∃ e v
  --   (c , d) ← stren-∃ e₁ v
  --   just (_ , (cong₂ (λ f g → E.backslide f g x x₁) b d))
  -- stren-∃ (bin x e e₁) v = do
  --   (a , b) ← stren-∃ e v
  --   (c , d) ← stren-∃ e₁ v
  --   just (_ , (cong₂ (bin x) b d))
  -- stren-∃ (scaledown x e) v =
  --   mmap (λ (a , b) → _ , (cong (scaledown x) b)) (stren-∃ e v)
  -- stren-∃ (un x e) v =
  --   mmap (λ (a , b) → _ , (cong (un x) b)) (stren-∃ e v)
  -- stren-∃ (maximum e) v =
  --   mmap (λ (a , b) → _ , (cong E.maximum b)) (stren-∃ e (there v))
  -- stren-∃ (let′ e e₁) v = do
  --   (a , b) ← stren-∃ e v
  --   (c , d) ← stren-∃ e₁ (there v)
  --   just (_ , (cong₂ let′ b d))

  -- stren? : (e : E Γ is) (v : ip ∈ Γ)
  --   → Dec (∃ λ (z : E (Γ / v) is) → e ≡ wk (wk-/ v) z)
  -- stren? (var x) v = map′ f g (strenv? v x) where
  --   f : _
  --   f (a , b) = _ , cong var b
  --   g : _
  --   g (var x , refl) = x , refl
  -- stren? zero v = yes (zero , refl)
  -- stren? one v = yes (one , refl)
  -- stren? (imaps e) v = map′ f g (stren? e (there v)) where
  --   f : _
  --   f (a , b) = _ , (cong imaps b)

  --   g : _
  --   g (imaps a , refl) = a , refl
  -- stren? {Γ = Γ} (imap e) v = map′ f g (stren? e (there v)) where
  --   f : _
  --   f (a , b) = _ , (cong imap b)

  --   g : ∃ (λ z → imap e ≡ wk (wk-/ v) z) →
  --     ∃ (λ z → e ≡ wk (wk-/ (there v)) z)
  --   g (a , b) = {!   !}
  -- stren? e v = {!   !}

  -- ∃stren : (e : E Γ is) (v : ip ∈ Γ)
  --   → Maybe (∃ λ (z : E (Γ / v) is) → IsE₂ e z)
  -- ∃stren (var x) v =
  --   mmap (λ (a , b , c) → _ , _ , c) (∃strenv v x)
  -- ∃stren zero v = just {!   !}
  -- ∃stren e v = {!   !}

  -- strenv-eq-eq : ∀ (x : is ∈ Γ) (z : ip ∈ (Γ / x))
  --   → strenv x (wkv (wk-/ x) z) ≡ just z
  -- strenv-eq-eq v₀ z = cong just (wkv-at-eq z)
  -- strenv-eq-eq (there x) v₀ = refl
  -- strenv-eq-eq (there x) (there z)
  --   with (strenv x (wkv (wk-/ x) z)) | (strenv-eq-eq x z)
  -- ... | just a | b = cong just (cong there (just-injective b))

  -- strenv-eq-eq : ∀ (x : is ∈ Γ) {y : ip ∈ Γ} {z : ip ∈ (Γ / x)}
  --   → strenv x y ≡ just z → y ≡ wkv (wk-/ x) z
  -- strenv-eq-eq v₀ {there y} {z} refl = cong there (sym (wkv-at-eq z))
  -- strenv-eq-eq (there x) {v₀} {v₀} eq = refl
  -- strenv-eq-eq (there x) {there y} {v₀} eq with (strenv x y) | eq
  -- ... | just a | ()
  -- ... | nothing | ()
  -- strenv-eq-eq (there x) {there y} {there z} eq =
  --   cong there (strenv-eq-eq x (strenv-inj₂ x eq))

  -- stren-eq-eq : ∀ {Γ is ip} (x : is ∈ Γ) {y : E Γ ip} {z : E (Γ / x) ip}
  --   → stren y x ≡ just z → IsE₂ y z
  -- stren-eq-eq {_} {_} {ix s} x {var y} {var z} eq = z , refl
  -- stren-eq-eq {_} {_} {ar s} x {var y} {z} eq = {!   !}
  -- stren-eq-eq {_} {_} {ar s} x {y} {z} eq = {!   !}

  -- stren-eq-eq : ∀ {Γ is ip} (x : is ∈ Γ) {y : E Γ ip} {z : E (Γ / x) ip}
  --     → stren y x ≡ just z → y ≡ wk (wk-/ x) z
  -- stren-eq-eq x {var y} {var z} eq =
  --   cong var (strenv-eq-eq x (var-stren-strenv x eq))
  -- -- stren-eq-eq x {𝟘} {𝟘} eq = refl
  -- -- stren-eq-eq x {one} {one} eq = refl
  -- -- stren-eq-eq {_} {_} {ar s} x {var} {zero} eq = {! eq  !}
  -- -- stren-eq-eq {_} {_} {ar s} (there x) {var (there y)} {z} eq = {!   !}
  -- stren-eq-eq {_} {_} {ar s} x {y} {z} eq with (stren y x) | z | eq | y
  -- ... | just a | b | eq' | d = {!   !}


  -- stren-eq-eq v₀ {var (there y)} {var z} refl =
  --   cong (λ x → var (there x)) (sym (wkv-at-eq y))
  -- stren-eq-eq (there x) {var v₀} {var z} eq with (just-injective eq)
  -- ... | refl = refl
  -- stren-eq-eq (there x) {var (there y)} {var v₀} eq = {!   !}
  -- stren-eq-eq (there x) {var (there y)} {var (there z)} eq = {!   !}
  -- stren-eq-eq x {y} {z} eq = {!   !}
