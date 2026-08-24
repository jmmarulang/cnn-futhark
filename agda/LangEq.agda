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
  isVar (argmax pf e) = no λ ()

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
  -- isZero (argmax pf e) = no λ ()

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
  -- isOne (argmax pf e) = no λ ()

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
  -- isImap (argmax pf e) = no λ { (_ , _ , refl , _ , ()) }


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
  -- isImaps (argmax pf e) = no λ ()

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
  -- isZeroBut (argmax pf e) = no λ ()

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
  -- isSels (argmax pf e) s = no λ { (refl , _ , _ , ()) }

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
  -- isSel (argmax pf e) = no λ { (_ , _ , _ , ()) }

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
  -- isImapb (argmax pf e) = no λ { (_ , _ , _ , _ , ()) }

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
  -- isSelb (argmax pf e) = no λ { (_ , _ , _ , _ , _ , ()) }

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
  -- isSum (argmax pf e) = no λ ()

  isArgmax : (e : E Γ (ix p)) → Dec (∃ λ p → ∃₂ λ pf t → e ≡ argmax {p = p} pf t)
  isArgmax (var x) = no λ ()
  isArgmax (argmax pf e) = yes (_ , pf , e , refl)

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
  -- isSlide (argmax pf e) = no λ ()

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
  -- isBackslide (argmax pf e) = no λ ()

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
  -- isUn (argmax pf e) = no λ ()

  un-inj : {a b : E Γ (ar s)} → {x y : Uop} → (un x a ≡ un y b) → (x ≡ y)
  un-inj {a = a} {b = b} {x = x} {y = y} refl = refl

  isInv : (e : E Γ (ar s)) → Dec (∃ λ t → e ≡ 𝟙/ t)
  isInv e with (isUn e)
  ... | no a = no (λ z → a (inverse , z))
  ... | yes (x , e , refl) with x ≟ᵘ inverse
  ... | yes refl = yes (e , refl)
  ... | no a = no λ (b , c) → a (un-inj c)

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
  -- isBin (argmax pf e) = no λ ()

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
  -- isScaledown (argmax pf e) = no λ ()

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
  -- isLet (argmax pf e) = no λ ()

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
  argmax {p = p} pf e ≟ᵉ u with isArgmax u
  ... | no _ = nothing
  ... | yes (p′ , pf′ , u , refl) with p ≟ˢ p′
  ... | no _ = nothing
  ... | yes refl with e ≟ᵉ u
  ... | just refl rewrite (suc≈-uniq pf pf′) = just refl
  ... | nothing = nothing

  -- test : (e : E (Γ ▹ ix s)) → inv (sum e)

  e-eq? : (a : E Γ is) (b : E Γ ip) → Maybe (Σ (is ≡ ip) λ pp → subst (E Γ) pp a ≡ b)
  e-eq? {is = is}{ip} a b with is ≟ⁱ ip
  ... | no ¬p = nothing
  ... | yes refl = a ≟ᵉ b >>= just ∘ (refl ,_)

  open import Data.Unit
  open import Data.Empty
  open import Data.Maybe renaming (map to mmap)
  open import Data.Maybe.Properties
  open import Data.Nat using (ℕ; zero; suc; _+_)
  open WkSub hiding (_∙ˢ_)

  -- chain-selx? : E Γ is → ip ∈ Γ → ℕ

  count-sels : E Γ is → ip ∈ Γ → ℕ
  count-sels (sels e e₁) v with isVar e
  ... | no a = count-sels e v
  ... | yes (w , refl) with eq? w v
  ... | veq = 1
  ... | _ = 0
  count-sels (sel e e₁) v = count-sels e v
  count-sels (selb x e e₁) v = count-sels e v
  count-sels (imaps e) v = count-sels e (there v)
  count-sels (imap e) v = count-sels e (there v)
  count-sels (imapb x e) v = count-sels e (there v)
  count-sels (sum e) v = count-sels e (there v)
  count-sels (argmax pf e) v = count-sels e v
  count-sels (zero-but i j e) v = count-sels e v
  count-sels (slide i x e x₁) v = count-sels e v
  count-sels (backslide i e x x₁) v = count-sels e v
  count-sels (bin x e e₁) v = (count-sels e v) + (count-sels e₁ v)
  count-sels (scaledown x e) v = count-sels e v
  count-sels (let′ e e₁) v = count-sels e v + count-sels e₁ (there v)
  count-sels (un x e) v = count-sels e v
  count-sels e v = 0

  -- count-selv : E Γ is → ip ∈ Γ → ℕ
  -- count-selv (sels e e₁) v with isVar e
  -- ... | no a = count-selv e v
  -- ... | yes (w , refl) with eq? w v
  -- ... | veq = 1
  -- ... | _ = 0
  -- count-selv (sel e e₁) v with isVar e
  -- ... | no a = count-selv e v
  -- ... | yes (w , refl) with eq? w v
  -- ... | veq = 1
  -- ... | _ = 0
  -- count-selv (selb x e e₁) v with isVar e
  -- ... | no a = count-selv e v
  -- ... | yes (w , refl) with eq? w v
  -- ... | veq = 1
  -- ... | _ = 0
  -- count-selv (imaps e) v = count-selv e (there v)
  -- count-selv (imap e) v = count-selv e (there v)
  -- count-selv (imapb x e) v = count-selv e (there v)
  -- count-selv (sum e) v = count-selv e (there v)
  -- count-selv (argmax pf e) v = count-selv e (there v)
  -- count-selv (zero-but i j e) v = count-selv e v
  -- count-selv (slide i x e x₁) v = count-selv e v
  -- count-selv (backslide i e x x₁) v = count-selv e v
  -- count-selv (bin x e e₁) v = (count-selv e v) + (count-selv e₁ v)
  -- count-selv (scaledown x e) v = count-selv e v
  -- count-selv (let′ e e₁) v = count-selv e v + count-selv e₁ (there v)
  -- count-selv (un x e) v = count-selv e v
  -- count-selv e v = 0

  -- test-count : ℕ
  -- test-count = count-selv {Γ = ε ▹ ar unit ▹ ix unit} {is = ar unit} (sels (var v₁) (var v₀) ⊞ sels (var v₁) (var v₀)) v₁

  inline : E Γ is → E Γ is
  inline e = norm-lets (inline' e) where
    inline' : E Γ is → E Γ is
    inline' (var x) = var x
    inline' 𝟘 = 𝟘
    inline' 𝟙 = 𝟙
    inline' (imaps e) = imaps (inline' e)
    inline' (sels e e₁) = sels (inline' e) (inline' e₁)
    inline' (imap e) = imap (inline e)
    inline' (sel e e₁) = sel (inline' e) (inline' e₁)
    inline' (imapb x e) = E.imapb x (inline' e)
    inline' (selb x e e₁) = E.selb x (inline' e) (inline' e₁)
    inline' (sum e) = E.sum (inline' e)
    inline' (zero-but e e₁ e₂) = (zero-but (inline' e) (inline' e₁) (inline' e₂))
    inline' (slide e x e₁ x₁) = E.slide (inline' e) x (inline' e₁) x₁
    inline' (backslide e e₁ x x₁) = E.backslide (inline' e) (inline' e₁) x x₁
    inline' (bin x e e₁) = bin x (inline' e) (inline' e₁)
    inline' (scaledown x e) = scaledown x (inline' e)
    inline' (un x e) = un x (inline' e)
    inline' (argmax pf e) = argmax pf (inline' e)
    inline' (let′ e e₁) with a ← (inline' e₁) | count-uses a v₀ | count-sels a v₀ | e
    ... | 0 | _ | _ = sub a (sub-id ▹ (inline' e))
    ... | _ | _ | var v = sub a (sub-id ▹ (var v))
    ... | _ | _ | zero = sub a (sub-id ▹ zero)
    ... | _ | _ | one = sub a (sub-id ▹ one)
    ... | 1 | 0 | _ = sub a (sub-id ▹ inline' e)
    ... | 1 | 1 | _ = sub a (sub-id ▹ inline' e)
    ... | _ | _ | _ = let′ (inline' e) a

  -- _⊆?_ : (Γ Δ : Ctx) → Dec (Γ ⊆ Δ)
  -- ε ⊆? Δ = yes ⊆-ε
  -- (Γ ▹ x) ⊆? ε = no λ ()
  -- (Γ ▹ x) ⊆? (Δ ▹ y) with (x ≟ⁱ y)
  -- ... | yes refl = map′ (λ s → keep s) ⊆-inj (Γ ⊆? Δ)
  -- ... | no f = no (λ s → {!   !})


  -- IsE₂ : ∀ {Γ' Γ is ip} (x : E Γ' is) (e : E Γ ip) → Set
  -- IsE₂ {_} {Γ} {_} {_} (var x) e = (∃ λ v → e ≡ var v)
  -- IsE₂ {_} {Γ} {ar s} {ix q} x e = ⊥
  -- IsE₂ {_} {Γ} {ar s} {ar q} 𝟘 e = (e ≡ zero)
  -- IsE₂ {_} {Γ} {ar s} {ar q} 𝟙 e = (e ≡ one)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (imaps x) e = (∃ λ u → e ≡ imaps u)
  -- IsE₂ {_} {Γ} {ar s} {ar p} (sels x x₁) e = (Σ (p ≡ []) λ eq → ∃₂ λ t u → subst (E Γ ∘ ar) eq e ≡ sels {s = s} t u)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (imap x) e = (∃₂ λ s p → Σ (s L.++ p ≡ q) λ eq → ∃ λ u → subst (E Γ ∘ ar) (sym eq) e ≡ imap {s = s} u)
  -- IsE₂ {_} {Γ} {ar s} {ar p} (sel x x₁) e =  (∃ λ s → ∃₂ λ t u → e ≡ sel {s = s}{p} t u)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (E.imapb x x₁) e = (∃₂ λ s p → Σ (s * p ≈ q) λ pf → ∃ λ t → e ≡ E.imapb pf t)
  -- IsE₂ {_} {Γ} {ar s} {ar p} (E.selb x x₁ x₂) e = (∃₂ λ s q → Σ (s * p ≈ q) λ pf → ∃₂ λ t u → e ≡ E.selb pf t u)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (E.sum x) e = (∃₂ λ s t → e ≡ E.sum {s = s} t)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (zero-but x x₁ x₂) e = (∃₂ λ s i → ∃₂ λ j u → e ≡ zero-but {s = s} i j u)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (E.slide x x₁ x₂ x₃) e = (∃₂ λ s′ p′ → ∃₂ λ r′ t → ∃₂ λ x′ t₁ → ∃ λ x₁ → e ≡ E.slide {s = s′}{p′}{r′} t x′ t₁ x₁)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (E.backslide x x₁ x₂ x₃) e = (∃₂ λ s′ u′ → ∃₂ λ p′ t → ∃₂ λ t₁ x → ∃ λ x₁ → e ≡ E.backslide {s = s′}{u = u′}{p = p′} t t₁ x x₁)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (bin x x₁ x₂) e = (∃₂ λ o t → ∃ λ t₁ → e ≡ bin o t t₁)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (scaledown x x₁) e = (∃₂ λ x t  → e ≡ scaledown x t)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (let′ x x₁) e = (∃₂ λ s′ t → ∃ λ t₁ → e ≡ let′ {s = s′} t t₁)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (un x x₁) e = (∃ λ t → ∃ λ t₁ → e ≡ un t t₁)
  -- IsE₂ {_} {Γ} {ar s} {ar q} (maximum x) e = (∃₂ λ s t → e ≡ maximum {s = s} t)
  -- -- IsE₂ {_} {Γ} {ar s} {ar q} (var x) e = (∃ λ v → e ≡ var v)

  -- isE₂ : ∀ {Γ' Γ is} (x : E Γ' is) (e : E Γ is) → Dec (IsE₂ x e)
  -- isE₂ {_} {_} {ar s} 𝟘 e = isZero e
  -- isE₂ {_} {_} {ar s} 𝟙 e = isOne e
  -- isE₂ {_} {_} {ar s} (imaps x) e = isImaps e
  -- isE₂ {_} {_} {ar s} (sels x x₁) e = isSels e s
  -- isE₂ {_} {_} {ar s} (imap x) e = isImap e
  -- isE₂ {_} {_} {ar s} (sel x x₁) e = isSel e
  -- isE₂ {_} {_} {ar s} (E.imapb x x₁) e = isImapb e
  -- isE₂ {_} {_} {ar s} (E.selb x x₁ x₂) e = isSelb e
  -- isE₂ {_} {_} {ar s} (E.sum x) e = isSum e
  -- isE₂ {_} {_} {ar s} (zero-but x x₁ x₂) e = isZeroBut e
  -- isE₂ {_} {_} {ar s} (E.slide x x₁ x₂ x₃) e = isSlide e
  -- isE₂ {_} {_} {ar s} (E.backslide x x₁ x₂ x₃) e = isBackslide e
  -- isE₂ {_} {_} {ar s} (bin x x₁ x₂) e = isBin e
  -- isE₂ {_} {_} {ar s} (scaledown x x₁) e = isScaledown e
  -- isE₂ {_} {_} {ar s} (let′ x x₁) e = isLet e
  -- isE₂ {_} {_} {ar s} (un x x₁) e = isUn e
  -- isE₂ {_} {_} {ar s} (maximum x) e = isMaximum e
  -- isE₂ {_} {_} {ix s} (var x) e = isVar e
  -- isE₂ {_} {_} {ar s} (var x) e = isVar e

