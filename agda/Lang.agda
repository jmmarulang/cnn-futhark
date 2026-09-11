{-# OPTIONS  --backtracking-instance-search #-}



-- {-# OPTIONS --warn=noUserWarning #-}
module _ where
module _ where
  open import Data.Nat using (ℕ; zero; suc)
  open import Data.List as L using (List; []; _∷_)
  open import Ar hiding (sum; slide; backslide; imapb; selb)
  open import Relation.Binary.PropositionalEquality
  open import Relation.Nullary
  open import Function
  infixl 15 _▹_

  cong₃ : {X Y Z W : Set} (f : X → Y → Z → W) → ∀ {x x₁ y y₁ z z₁}
        → x ≡ x₁ → y ≡ y₁ → z ≡ z₁ → f x y z ≡ f x₁ y₁ z₁
  cong₃ _ refl refl refl = refl

  data IS : Set where
    ix  : S → IS
    ar  : S → IS

  data Ctx : Set where
    ε    : Ctx
    _▹_  : Ctx → IS → Ctx

  variable
    Γ Δ Ξ Ψ : Ctx
    is ip iq ir : IS

  data _∈_ : IS → Ctx → Set where
    here  : is ∈ (Γ ▹ is)
    there : is ∈ Γ → is ∈ (Γ ▹ ip)

  pattern v₀ = here
  pattern v₁ = there v₀
  pattern v₂ = there v₁
  pattern v₃ = there v₂
  pattern v₄ = there v₃
  pattern v₅ = there v₄
  pattern v₆ = there v₅
  pattern v₇ = there v₆
  pattern v₈ = there v₇
  pattern v₉ = there v₈
  pattern v₁₀ = there v₉
  pattern v₁₁ = there v₁₀
  pattern v₁₂ = there v₁₁
  pattern v₁₃ = there v₁₂

  v-inj : ∀ {ip} {x y : is ∈ Γ} → (there {ip = ip} x ≡ there y) → (x ≡ y)
  v-inj refl = refl

  -- We only use this for variable comparison.
  _/_ : (Γ : Ctx) → is ∈ Γ → Ctx
  (Γ ▹ x) / here = Γ
  (Γ ▹ x) / there v = (Γ / v) ▹ x

  wkv-/ : (v : is ∈ Γ) → ip ∈ (Γ / v) → ip ∈ Γ
  wkv-/ here w = there w
  wkv-/ (there v) here = here
  wkv-/ (there v) (there w) = there (wkv-/ v w)

  data Eq : is ∈ Γ → ip ∈ Γ → Set where
    veq : {x : is ∈ Γ} → Eq x x
    neq : (x : is ∈ Γ) → (y : ip ∈ (Γ / x)) → Eq x (wkv-/ x y)

  eq? : (x : is ∈ Γ) → (y : ip ∈ Γ) → Eq x y
  eq? v₀ v₀ = veq
  eq? v₀ (there y) = neq v₀ y
  eq? (there x) v₀ = neq (there x) v₀
  eq? (there x) (there y) with eq? x y
  ... | veq = veq
  ... | neq .x y = neq (there x) (there y)


  unthere : {x y : is ∈ Γ} → there {ip = ip} x ≡ there y → x ≡ y
  unthere refl = refl

  neq-wkv : (x : is ∈ Γ) (y : is ∈ (Γ / x)) → x ≢ wkv-/ x y
  neq-wkv v₀ y = λ ()
  neq-wkv (there x) v₀ = λ ()
  neq-wkv (there x) (there y) p = (neq-wkv x y) (unthere p)

  infixl 10 _⊞_
  infixl 10 _⊟_
  infixl 15 _⊠_
  -- infixl 15 _⊔_

  unit : S
  unit = []

  data Kop : Set where
    zero-op one-op : Kop

  data Uop : Set where
      neg-op
        relu-op
        sqrt-op
        inv-op
        ind-op
        ln-op
        softmax-op
        : Uop
      scaledown-op : ℕ → Uop

  data Bop : Set where
    plus-op mul-op : Bop

  data Mop   : S → S → S → Set where
    imaps-op : Mop s unit s
    imap-op  : q ≡ s ⊗ p → Mop s p q
    imapb-op : s * p ≈ q → Mop s p q
    sum-op   : Mop s p p

  data Sop  : S → S → S → Set where
    sels-op : Sop s s unit
    sel-op  : q ≡ s ⊗ p → Sop q s p
    selb-op : s * p ≈ q → Sop q s p

  data E : Ctx → IS → Set where
    var        : is ∈ Γ → E Γ is

    kop        : Kop → E Γ (ar s)
    uop        : Uop → E Γ (ar s) → E Γ (ar s)
    bop        : Bop → E Γ (ar s) → E Γ (ar s) → E Γ (ar s)

    mop        : Mop s p q → E (Γ ▹ ix s) (ar p) → E Γ (ar q)
    sop        : Sop s p q → E Γ (ar s) → E Γ (ix p) → E Γ (ar q)

    zero-but   : E Γ (ix s) → E Γ (ix s) → E Γ (ar p) → E Γ (ar p)

    let′       : E Γ (ar s) → E (Γ ▹ ar s) (ar p) → E Γ (ar p)

  pattern 𝟙 {s = s} = kop {s = s} one-op
  pattern 𝟘 {s = s} = kop {s = s} zero-op

  pattern ⊟_   a = uop neg-op a
  pattern √ a = uop sqrt-op a
  pattern 𝟙/   a = uop inv-op a
  pattern relu a = uop relu-op a
  pattern ln   a = uop ln-op a
  pattern 𝕚+   a = uop ind-op a
  pattern ℙ    a = uop softmax-op a
  pattern scaledown n a = uop (scaledown-op n) a

  pattern _⊞_ a b = bop plus-op a b
  pattern _⊠_ a b = bop mul-op a b

  pattern imaps {s = s} {p = p} {q = q} a =
    mop {s = s} {p = p} {q = q} imaps-op a
  pattern imap′ {s = s} {p = p} {q = q} a b =
    mop {s = s} {p = p} {q = q} (imap-op a) b
  pattern imapb {s = s} {p = p} {q = q} a b =
    mop {s = s} {p = p} {q = q} (imapb-op a) b
  pattern sum {s = s} {p = p} {q = q} a =
    mop {s = s} {p = p} {q = q} sum-op a

  pattern sels {s = s} {p = p} {q = q} a b =
    sop {s = s} {p = p} {q = q} sels-op a b
  pattern sel′ {s = s} {p = p} {q = q} a b c =
    sop {s = q} {p = s} {q = p} (sel-op a) b c
  pattern selb {s = s} {p = p} {q = q} a b c =
    sop {s = s} {p = p} {q = q} (selb-op a) b c

  imap : E (Γ ▹ ix s) (ar p) → E Γ (ar (s ⊗ p))
  imap e = imap′ refl e

  sel : E Γ (ar (s ⊗ p)) → E Γ (ix s) → E Γ (ar p)
  sel e i = sel′ refl e i

  _⊟_ : ( a b : E Γ (ar s)) → E Γ (ar s)
  _⊟_ a b = a ⊞ ⊟ b

  _//_ : ( a b : E Γ (ar s)) → E Γ (ar s)
  _//_ a b = a ⊠ (𝟙/ b)

  𝕚0- : (E Γ (ar s)) → E Γ (ar s)
  𝕚0- a = 𝟙 ⊟ 𝕚+ a

  𝕚0+ : (E Γ (ar s)) → E Γ (ar s)
  𝕚0+ a = 𝕚0- (⊟ a)

  𝕚≤ : (E Γ (ar s)) → (E Γ (ar s)) → E Γ (ar s)
  𝕚≤ a b = 𝕚0+ (a ⊟ b)

  𝟚 : E Γ (ar s)
  𝟚 = 𝟙 ⊞ 𝟙

  var-inj : ∀ {x y : is ∈ Γ} → (var x ≡ var y) → (x ≡ y)
  var-inj refl = refl

  scaledown-inj : ∀ {x y} → scaledown-op x ≡ scaledown-op y → (x ≡ y)
  scaledown-inj refl = refl

module WkSub where
  open import Data.Nat using (ℕ; zero; suc; _+_)
  open import Relation.Binary.PropositionalEquality
  open import Function
  -- -- open import Ar hiding (sum; slide; backslide; map ; imapb; selb)

  data _⊆_ : Ctx → Ctx → Set where
    ε    : ε ⊆ ε
    skip : Γ ⊆ Δ → Γ ⊆ (Δ ▹ is)
    keep : Γ ⊆ Δ → (Γ ▹ is) ⊆ (Δ ▹ is)

  wkv : Γ ⊆ Δ → is ∈ Γ → is ∈ Δ
  wkv (skip s) v = there (wkv s v)
  wkv (keep s) v₀ = v₀
  wkv (keep s) (there v) = there (wkv s v)

  wk : Γ ⊆ Δ → E Γ is → E Δ is
  wk s (var x) = var (wkv s x)
  wk s (kop x) = kop x
  wk s (uop x e) = uop x (wk s e)
  wk s (bop x e e₁) = bop x (wk s e) (wk s e₁)
  wk s (mop x e) = mop x (wk (keep s) e)
  wk s (sop x e e₁) = sop x (wk s e) (wk s e₁)
  wk s (zero-but e e₁ e₂) = zero-but (wk s e) (wk s e₁) (wk s e₂)
  wk s (let′ e e₁) = let′ (wk s e) (wk (keep s) e₁)

  _∙ʷ_ : Δ ⊆ Ψ → Γ ⊆ Δ → Γ ⊆ Ψ
  s ∙ʷ ε = s
  skip s ∙ʷ skip p = skip (s ∙ʷ skip p)
  keep s ∙ʷ skip p = skip s ∙ʷ p
  skip s ∙ʷ keep p = skip (s ∙ʷ keep p)
  keep s ∙ʷ keep p = keep (s ∙ʷ p)

  ⊆-eq : Γ ⊆ Γ
  ⊆-eq {ε} = ε
  ⊆-eq {Γ ▹ x} = keep ⊆-eq

  ⊆-ε : ε ⊆ Γ
  ⊆-ε {ε} = ε
  ⊆-ε {Γ ▹ x} = skip ⊆-ε

  ⊆-inj : (Γ ▹ is) ⊆ (Δ ▹ ip) → Γ ⊆ Δ
  ⊆-inj (skip s) = s ∙ʷ skip ⊆-eq
  ⊆-inj (keep s) = s ∙ʷ ⊆-eq

  open import Data.Empty


  _↑ : E Γ is → E (Γ ▹ ip) is
  _↑ = wk (skip ⊆-eq)

  wk-/ : (v : is ∈ Γ) → (Γ / v) ⊆ Γ
  wk-/ v₀ = skip ⊆-eq
  wk-/ (there v) = keep (wk-/ v)

  data Sub (Γ : Ctx) : Ctx → Set where
    ε   : Sub Γ ε
    _▹_ : Sub Γ Δ → E Γ is → Sub Γ (Δ ▹ is)

  wks : Sub Γ Δ → Γ ⊆ Ψ → Sub Ψ Δ
  wks ε p = ε
  wks (s ▹ x) p = (wks s p) ▹ wk p x

  sdrop : Sub Γ Δ → Sub (Γ ▹ is) Δ
  sdrop s = wks s (skip ⊆-eq)

  skeep : Sub Γ Δ → Sub (Γ ▹ is) (Δ ▹ is)
  skeep s = sdrop s ▹ var v₀

  subv : Sub Γ Δ → is ∈ Δ → E Γ is
  subv (s ▹ x) v₀ = x
  subv (s ▹ x) (there v) = subv s v

  sub-id : Sub Γ Γ
  sub-id {ε} = ε
  sub-id {Γ ▹ x} = skeep sub-id

  sub : E Δ is → Sub Γ Δ → E Γ is
  sub (var x) s = subv s x
  sub (kop x) s = kop x
  sub (uop x e) s = uop x (sub e s)
  sub (bop x e e₁) s = bop x (sub e s) (sub e₁ s)
  sub (mop x e) s = mop x (sub e (skeep s))
  sub (sop x e e₁) s = sop x (sub e s) (sub e₁ s)
  sub (zero-but e e₁ e₂) s = zero-but (sub e s) (sub e₁ s) (sub e₂ s)
  sub (let′ e e₁) s = let′ (sub e s) (sub e₁ (skeep s))

  _∙ˢ_ : Sub Δ Ψ → Sub Γ Δ → Sub Γ Ψ
  ε ∙ˢ t = ε
  (s ▹ x) ∙ˢ t = (s ∙ˢ t) ▹ sub x t

  -- All kinds of theorems
  wkv-at-eq : (v : is ∈ Γ) → wkv ⊆-eq v ≡ v
  wkv-at-eq v₀ = refl
  wkv-at-eq (there v) = cong there (wkv-at-eq v)

  subv-wks : (v : is ∈ Γ) (s : Sub Δ Γ) (w : Δ ⊆ Ψ) → subv (wks s w) v ≡ wk w (subv s v)
  subv-wks v₀ (s ▹ x) w = refl
  subv-wks (there v) (s ▹ x) w = subv-wks v s w

  subv-sdrop : (v : is ∈ Γ) (s : Sub Δ Γ) → subv (sdrop {is = ip} s) v ≡ (subv s v) ↑
  subv-sdrop v₀ (s ▹ x) = refl
  subv-sdrop (there v) (s ▹ x) = subv-wks v s _

  subv-at-id : (v : is ∈ Γ) → subv sub-id v ≡ var v
  subv-at-id v₀ = refl
  subv-at-id {is} {.(Γ ▹ ip)} (there {is = .is} {Γ = Γ} {ip = ip} v)
    rewrite subv-sdrop {ip = ip} v sub-id | subv-at-id v = cong (var ∘′ there) (wkv-at-eq v)

  sub-at-id : (e : E Γ is) → sub e sub-id ≡ e
  sub-at-id (var x) = subv-at-id x
  sub-at-id (kop x) = refl
  sub-at-id (uop x e) = cong (uop x) (sub-at-id e)
  sub-at-id (bop x e e₁) = cong₂ (bop x) (sub-at-id e) (sub-at-id e₁)
  sub-at-id (mop x e) = cong (mop x) (sub-at-id e)
  sub-at-id (sop x e e₁) = cong₂ (sop x) (sub-at-id e) (sub-at-id e₁)
  sub-at-id (zero-but e e₁ e₂) rewrite (sub-at-id e) | sub-at-id e₁ | sub-at-id e₂ = refl
  sub-at-id (let′ e e₁) = cong₂ let′ (sub-at-id e) (sub-at-id e₁)

  sub-ε : (e : E ε is) → sub e ε ≡ e
  sub-ε e = sub-at-id e

  sub-swap : Sub (Γ ▹ is ▹ ip) (Γ ▹ ip ▹ is)
  sub-swap = (sdrop (sdrop sub-id) ▹ var v₀) ▹ var (there v₀)

  -- We are not really using this, but this is a useful function to have.
  open import Data.Maybe
  open import Data.Maybe.Properties
  open import Data.Product hiding (map)

  strenv-∃ : (x : is ∈ Γ) (y : ip ∈ Γ)
    → Maybe (∃ (λ (z : ip ∈ (Γ / x)) → y ≡ wkv (wk-/ x) z))
  strenv-∃ v₀ v₀ = nothing
  strenv-∃ v₀ (there y) = just (y , (cong there (sym (wkv-at-eq y))))
  strenv-∃ (there x) v₀ = just (v₀ , refl)
  strenv-∃ (there x) (there y) =
    map (λ (a , eq) → there a , (cong there eq)) (strenv-∃ x y)

  stren-∃ : (e : E Γ is) (v : ip ∈ Γ)
    → Maybe (∃ λ (z : E (Γ / v) is) → e ≡ wk (wk-/ v) z)
  stren-∃ (var x) v = map (λ (a , b) → _ , (cong var b)) (strenv-∃ v x)
  stren-∃ (kop x) v = just (kop x , refl)
  stren-∃ (uop x e) v = map (λ (a , b) → _ , (cong (uop x) b)) (stren-∃ e v)
  stren-∃ (bop x e e₁) v = do
    (_ , a) ← stren-∃ e v
    (_ , b) ← stren-∃ e₁ v
    just (_ , (cong₂ (bop x) a b))
  stren-∃ (mop x e) v = map (λ (a , b) → _ , (cong (mop x) b)) (stren-∃ e (there v))
  stren-∃ (sop x e e₁) v = do
    (_ , a) ← stren-∃ e v
    (_ , b) ← stren-∃ e₁ v
    just (_ , (cong₂ (sop x) a b))
  stren-∃ (zero-but e e₁ e₂) v = do
    (a , b) ← stren-∃ e v
    (c , d) ← stren-∃ e₁ v
    (e , f) ← stren-∃ e₂ v
    just (_ , cong₃ zero-but b d f)
  stren-∃ (let′ e e₁) v = do
    (a , b) ← stren-∃ e v
    (c , d) ← stren-∃ e₁ (there v)
    just (_ , (cong₂ let′ b d))

  stren : (e : E Γ is) (v : ip ∈ Γ)
    → Maybe (E (Γ / v) is)
  stren e v = do
    (a , _) ← (stren-∃ e v)
    just a

  norm-lets : E Γ is → E Γ is
  norm-lets (var x) = var x
  norm-lets (kop x) = kop x
  norm-lets (uop x e) = uop x (norm-lets e)
  norm-lets (bop x e e₁) = bop x (norm-lets e) (norm-lets e₁)
  norm-lets (mop x e) = mop x (norm-lets e)
  norm-lets (sop x e e₁) = sop x (norm-lets e) (norm-lets e₁)
  norm-lets (zero-but e e₁ e₂) = zero-but (norm-lets e) (norm-lets e₁) (norm-lets e₂)
  norm-lets (let′ e e₁) = maybe id (let′ (norm-lets e) (norm-lets e₁)) (stren (norm-lets e₁) v₀)

  count-uses : E Γ is → ip ∈ Γ → ℕ
  count-uses (var x) v with eq? x v
  ... | veq = 1
  ... | _ = 0
  count-uses (kop x) v = 0
  count-uses (uop x e) v = count-uses e v
  count-uses (bop x e e₁) v = count-uses e v + count-uses e₁ v
  count-uses (mop x e) v = count-uses e (there v)
  count-uses (sop x e e₁) v = count-uses e v + count-uses e₁ v
  count-uses (zero-but e e₁ e₂) v = count-uses e v + count-uses e₁ v + count-uses e₂ v
  count-uses (let′ e e₁) v = count-uses e v + count-uses e₁ (there v)

module Syntax where
  open import Data.List as L using (List; []; _∷_)
  open import Ar hiding (sum; imapb)

  -- Convenience functions when writing expressions in the DSL
  -- In some sense we are faking HOAS using instance resolution.
  data Prefix : (Γ Δ : Ctx) → Set where
    instance
      zero : Prefix Γ Γ
      suc  : ⦃ Prefix Γ Δ ⦄ → Prefix Γ (Δ ▹ is)

  -- A term that can be lifted into larger contexts
  GE : Ctx → IS → Set
  GE Γ is = ∀ {Δ} → ⦃ Prefix Γ Δ ⦄ → E Δ is

  -- A variable that can be lifted into larger contexts
  GVar : Ctx → IS → Set
  GVar Γ is = ∀ {Δ} → ⦃ p : Prefix Γ Δ ⦄ → is ∈ Δ

  -- Lift var
  V : is ∈ Γ → GVar Γ is
  V v ⦃ p = zero ⦄ = v
  V v ⦃ p = suc  ⦄ = there (V v)

  -- Use GE GVar and V to define HOAS-style imap, imaps, and impab
  Imap : ∀ {Γ}
       → (GE (Γ ▹ ix s) (ix s) → E (Γ ▹ ix s) (ar p))
       → E Γ (ar (s ⊗ p))
  --Imap f = imap (f λ {Δ} ⦃ p ⦄ → var (V v₀))
  Imap f = imap (f (var (V v₀)))

  Sum : ∀ {Γ}
       → (GE (Γ ▹ ix s) (ix s) → E (Γ ▹ ix s) (ar p))
       → E Γ (ar p)
  Sum f = sum (f λ {Δ} ⦃ p ⦄ → var (V v₀))

  Imaps : ∀ {Γ}
        → (GE (Γ ▹ ix s) (ix s) → E (Γ ▹ ix s) (ar unit))
        → E Γ (ar s)
  Imaps f = imaps (f λ {Δ} ⦃ p ⦄ → var (V v₀))

  Imapb : ∀ {Γ}
        → s * p ≈ q
        → (GE (Γ ▹ ix s) (ix s) → E (Γ ▹ ix s) (ar p))
        → E Γ (ar q)
  Imapb p f = imapb p (f λ {Δ} ⦃ p ⦄ → var (V v₀))

  Let-syntax : ∀ {Γ}
      → (E Γ (ar s))
      → (GE (Γ ▹ (ar s)) (ar s) → E (Γ ▹ (ar s)) (ar p))
      → E Γ (ar p)
  Let-syntax x f = let′ x (f λ {Δ} ⦃ p ⦄ → var (V v₀))

  infixl 3 Let-syntax
  syntax Let-syntax e (λ x → e') = Let x := e In e'

  -- Extend context with a list of types
  -- (List is a context that grows to the left)
  ext : Ctx → List IS → Ctx
  ext Γ [] = Γ
  ext Γ (x ∷ l) = ext (Γ ▹ x) l

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
  lfun : (l : List IS)  (Γ : Ctx) (is : IS) → Set
  lfun l Γ τ = lfunh l (E (ext Γ l) τ) (ext Γ l)

  -- Compute GE from the variable in the non-extended context
  lvar : ∀ l → is ∈ Γ → GE (ext Γ l) is
  lvar [] v = var (V v)
  lvar (x ∷ l) v = lvar l (there v)

  -- Apply function to the corresponding variables of the context
  Lcon : ∀ l is Γ → (f : lfun l Γ is) → E (ext Γ l) is
  Lcon []      is Γ f = f
  Lcon (x ∷ l) is Γ f = Lcon l is (Γ ▹ x) (f (lvar l v₀))

module Primitives where

  open import Data.List as L using (List; []; _∷_)
  open import Data.Nat as ℕ using (ℕ; zero; suc)
  open import Function using (_$_; it; _∋_)
  open import Relation.Binary.PropositionalEquality
  open import Ar hiding (slide; selb; swap; sum)
  open Syntax
  open WkSub

  fromPrefix : Prefix Γ Δ → Γ ⊆ Δ
  fromPrefix zero = ⊆-eq
  fromPrefix (suc ⦃ p ⦄) = skip (fromPrefix p)

  wkp : Prefix Γ Δ → E Γ is → E Δ is
  wkp p = wk (fromPrefix p)

  ⟨_⟩ : E Γ is → GE Γ is
  ⟨_⟩ t {Δ} ⦃ p ⦄ = wkp p t


  module Microgpt where

    open import Data.Product as Prod hiding (_<*>_)
    open import Data.List.Properties

    variable
      bs ah hd ed sl vo pr fd : S

    e-map : ∀ {Γ} (f : E (Γ ▹ ix p) (ar s) → E (Γ ▹ ix p) (ar q)) →
            E Γ (ar $ p ⊗ s) → E Γ (ar $ p ⊗ q)
    e-map {p = p} {s = s} f e = Imap {p} λ i → f (sel ⟨ e ⟩ i)

    tiles : ∀ {Γ} → E Γ (ar []) → E Γ (ar s)
    tiles x = Imaps (λ _ → ⟨ x ⟩)

    tile : ∀ {p s} {Γ} → E Γ (ar s) → E Γ (ar (p ⊗ s))
    tile x = Imap (λ _ → ⟨ x ⟩)

    iswap : ∀ {Γ} → E Γ (ar (s ⊗ u)) → E Γ (ar (u ⊗ s))
    iswap {s} {u} x = Imap {u} λ i → Imaps {s} λ j → sels (sel ⟨ x ⟩ j) i

    iswap3 : ∀ {Γ} → E Γ (ar (p ⊗ (s ⊗ u))) → E Γ (ar (s ⊗ (p ⊗ u)))
    iswap3 {p} {s} {u} x = Imap {s} λ i → Imap {p} λ j → sel (sel ⟨ x ⟩ j) i

    -- DANGER: implicit use of substitution
    iassᵣ : ∀ {s p q} {Γ} → E Γ (ar (s ⊗ (p ⊗ q))) → E Γ (ar ((s ⊗ p) ⊗ q))
    iassᵣ {s} {p} {q} e rewrite (⊗-ass {s} {p} {q}) = e
      -- Imap λ i → Imaps {q} λ j →
      -- sels (Imap {s} λ k → Imaps λ w → sels (sel (sel ⟨ e ⟩ k) w) j) i

    iassₗ : ∀ {s p q} {Γ} → E Γ (ar ((s ⊗ p) ⊗ q)) → E Γ (ar (s ⊗ (p ⊗ q)))
    iassₗ {s} {p} {q} e = Imap {s} λ i → Imap {p} λ j → Imaps λ k →
      sels (sel (Imaps λ w → sels (sel ⟨ e ⟩ w) k) i) j

    -- DANGER: implicit use of substitution
    imedial : ∀ {Γ} → E Γ (ar $ (s ⊗ p) ⊗ (q ⊗ r))
              → E Γ (ar $ (s ⊗ q) ⊗ (p ⊗ r))
    imedial {s} {p} {q} {r} e
      rewrite (⊗-ass {s} {q} {(p ⊗ r)}) | ⊗-ass {s} {p} {(q ⊗ r)} =
      Imap {s} λ i → Imap {q} λ j → Imap {p} λ k → Imaps {r} λ w →
        sels (sel (sel (sel ⟨ e ⟩ i) k) j) w

      -- Imap λ sqi → Imaps λ pri →
      --   sels (Imap {s} λ si → Imaps {q} λ qi →
      --     sels (Imap {p} λ pi → Imaps {r} λ ri →
      --       sels (sel (Imaps λ spi →
      --         sels (sel (sel ⟨ e ⟩ spi) qi) ri) si) pi) pri) sqi

    -- TODO : add biases
    linear : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar s) → E Γ (ar u)
    linear {u} {s} w x =
      Imaps {u} λ i → Sum {s} λ j → sels (sel ⟨ w ⟩ i ⊠ ⟨ x ⟩) j

    m-linear : ∀ {Γ} → E Γ (ar $ u ⊗ s) →
                 E Γ (ar $ p ⊗ s) → E Γ (ar $ p ⊗ u)
    m-linear {u} {s} {p} w e = e-map {p} (linear (w ↑)) e

    matmul : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (s ⊗ r)) → E Γ (ar (u ⊗ r))
    matmul {u} {s} {r} w1 w2 =
      Imap {u} λ i → Imaps {r} λ j → Sum {s} λ k →
      sels (sel ⟨ w1 ⟩ i) k ⊠ sels (sel ⟨ w2 ⟩ k) j

    m-matmul : ∀ {Γ} → E Γ (ar $ p ⊗ (u ⊗ s)) → E Γ (ar $ p ⊗ (s ⊗ r))
               → E Γ (ar $ p ⊗ (u ⊗ r))
    m-matmul {p} {u} a b = e-map {p} (matmul {u} $ sel ⟨ a ⟩ (var v₀)) b

    matmult : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (r ⊗ s)) → E Γ (ar (u ⊗ r))
    matmult {u} {s} {r} w1 w2 =
      Imap {u} λ i → Imaps {r} λ j → Sum {s} λ k →
      sels ((sel ⟨ w1 ⟩ i) ⊠ (sel ⟨ w2 ⟩ j)) k

    m-matmult : ∀ {Γ} → E Γ (ar $ p ⊗ (u ⊗ s)) → E Γ (ar $ p ⊗ (r ⊗ s))
               → E Γ (ar $ p ⊗ (u ⊗ r))
    m-matmult {p} {u} a b = e-map {p} (matmult {u} $ sel ⟨ a ⟩ (var v₀)) b

    rmsnorm : ∀ {Γ} → E Γ (ar s) → E Γ (ar s)
    rmsnorm {s = s} x =
      Let xx := x ⊠ x In
      Let ms := scaledown (len s) (Sum (λ i → sels xx i)) In
      Let scale := √ (ms ⊞ (scaledown 100000 𝟙)) In
      Imaps λ i → (sels ⟨ x ⟩ i) // scale

    m-rmsnorm : ∀ {Γ} → E Γ (ar (s ⊗ p)) → E Γ (ar (s ⊗ p))
    m-rmsnorm {s} {p} {Γ} e = e-map {s} rmsnorm e

    -- m-rmsnorm : ∀ {Γ} → E Γ (ar (p ⊗ s)) → E Γ (ar (p ⊗ s))
    -- m-rmsnorm {p} {s} {Γ} e =
    --   Let ee := e ⊠ e In
    --   Let ms := (Imaps {p} λ i → Sum λ j →
    --     sels (sel (scaledown (len s) ee) i) j) In
    --   Let scale := √ (ms ⊞ (scaledown 100000 𝟙)) In
    --   Imap {p} λ i → Imaps λ j → sels (sel ⟨ e ⟩ i) j // sels scale i

    record GPT-Params (Γ : Ctx) (vo ed sl fd : S) : Set₁ where
      field
        -- position embedding
        wpe : E Γ (ar (sl ⊗ ed))
        -- weights for queries, keys, values and outputs
        wqry wkey wval wout : E Γ (ar (ed ⊗ ed))
        -- up-projection
        wup : E Γ (ar (fd ⊗ ed))
        -- down projection
        wdown : E Γ (ar (ed ⊗ fd))
        -- output projection into vocabulary size
        wvoc : E Γ (ar (vo ⊗ ed))
        -- token embedding
        -- wte : E Γ (ar (vs ⊗ ed))

    open GPT-Params

    to-gptp : ∀ {Γ}
              (wpe : E Γ (ar (sl ⊗ ed)))
              (wqry wkey wval wout : E Γ (ar (ed ⊗ ed)) )
              (wup : E Γ (ar (fd ⊗ ed)))
              (wdown : E Γ (ar (ed ⊗ fd)))
              (wvoc : E Γ (ar (vo ⊗ ed)))
              → GPT-Params Γ vo ed sl fd
    to-gptp wpe wqry wkey wval wout wup wdown wvoc = record
      { wpe = wpe
      ; wqry = wqry
      ; wkey = wkey
      ; wval = wval
      ; wout = wout
      ; wup = wup
      ; wdown = wdown
      ; wvoc = wvoc
      }

    wk-gptp : Prefix Γ Δ → GPT-Params Γ vo ed sl fd → GPT-Params Δ vo ed sl fd
    wk-gptp pr p = record
      { wpe = wk (fromPrefix pr) (p .wpe)
      ; wqry = wk (fromPrefix pr) (p .wqry)
      ; wkey = wk (fromPrefix pr) (p .wkey)
      ; wval = wk (fromPrefix pr) (p .wval)
      ; wout = wk (fromPrefix pr) (p .wout)
      ; wup = wk (fromPrefix pr) (p .wup)
      ; wdown = wk (fromPrefix pr) (p .wdown)
      ; wvoc = wk (fromPrefix pr) (p .wvoc)
      }

    G-GPT-Params : Ctx → S → S → S → S → Set₁
    G-GPT-Params Γ vo ed sl fd =
      ∀ {Δ} → ⦃ Prefix Γ Δ ⦄ → GPT-Params Δ vo ed sl fd

    ⟨_⟩ₚ : GPT-Params Γ vo ed sl fd → G-GPT-Params Γ vo ed sl fd
    ⟨_⟩ₚ p {Δ} ⦃ pf ⦄  = wk-gptp pf p

    split-heads : ∀ {bs ah hd sl ed Γ} → ah * hd ≈ ed
      → E Γ (ar $ (bs ⊗ sl) ⊗ ed) → E Γ (ar $ (bs ⊗ sl) ⊗ (ah ⊗ hd))
    split-heads {bs = bs} {ah = ah} {sl = sl} pf e =
      e-map (λ x → Imap λ i → selb pf (x ↑) i) e

    merge-heads : ∀ {bs ah hd sl ed Γ} → ah * hd ≈ ed
      → E Γ (ar $ (bs ⊗ sl) ⊗ (ah ⊗ hd)) → E Γ (ar $ (bs ⊗ sl) ⊗ ed)
    merge-heads {bs = bs} {ah = ah} {sl = sl} pf e =
      e-map {bs ⊗ sl} (λ x → Imapb pf λ i → sel ⟨ x ⟩ i) $ e

    mask-attn-weights : ∀ {Γ} → E Γ (ar $ bs ⊗ (sl ⊗ sl))
                        → E Γ (ar $ (bs ⊗ ah) ⊗ (sl ⊗ sl))
                        → E Γ (ar $ (bs ⊗ ah) ⊗ (sl ⊗ sl))
    mask-attn-weights {bs} {ah = ah} mask w =
      -- DANGER: iassᵣ is inneficient. Opt should be able to take care of it
      w ⊞ iassᵣ {bs} (Imap {bs} λ i → tile {ah} (sel ⟨ mask ⟩ i))

    -- DANGER: Is this correct?
    gpt-softmax : ∀ {Γ} → E Γ (ar $ (bs ⊗ ah) ⊗ (sl ⊗ sl))
                  → E Γ (ar $ (bs ⊗ ah) ⊗ (sl ⊗ sl))
    gpt-softmax {bs} {ah} {sl} e =
      e-map {bs ⊗ ah} (λ x → e-map {sl} ℙ x) e

    multihead-attn : ∀ {Γ} (sc : ℕ) (mask : E Γ (ar $ bs ⊗ (sl ⊗ sl)))
                     (q k v : E Γ (ar $ (bs ⊗ ah) ⊗ (sl ⊗ hd)))
                     → E Γ (ar $ (bs ⊗ ah) ⊗ (sl ⊗ hd))
    multihead-attn {bs} {sl} {ah} sc mask q k v =
      Let w := m-matmult {bs ⊗ ah} {sl} q k In
      Let masked := mask-attn-weights {bs} {sl} ⟨ mask ⟩ (scaledown sc w) In
      Let sf := gpt-softmax {bs} {ah} {sl} masked
      In m-matmul {bs ⊗ ah} {sl} sf ⟨ v ⟩

    attn : ∀ {Γ}
           (pf : ah * hd ≈ ed) (sc : ℕ)
           (p : GPT-Params Γ vo ed sl fd)
           (mask : E Γ (ar $ bs ⊗ (sl ⊗ sl)))
           (h : E Γ (ar $ (bs ⊗ sl) ⊗ ed))
           → E Γ (ar $ (bs ⊗ sl) ⊗ ed)
    attn {ah} {hd} {ed} {vo} {sl} {fd} {bs} pf sc p mask h =
      Let q₀ := m-linear {ed} ⟨ p .wqry ⟩   h In
      Let k₀ := m-linear {ed} ⟨ p .wkey ⟩ ⟨ h ⟩ In
      Let v₀ := m-linear {ed} ⟨ p .wval ⟩ ⟨ h ⟩ In
      Let q₁ := split-heads {bs} pf q₀ In
      Let k₁ := split-heads {bs} pf k₀ In
      Let v₁ := split-heads {bs} pf v₀ In
      Let q := imedial {bs} q₁ In
      Let k := imedial {bs} k₁ In
      Let v := imedial {bs} v₁ In
      Let a₀ := multihead-attn {bs} {sl} sc ⟨ mask ⟩ q k v In
      Let a₁ := imedial {bs} {ah} {sl} a₀ In
      Let a := merge-heads {bs} pf a₁ In
      m-linear {ed} ⟨ p .wout ⟩ a

    mlp : ∀ {Γ}
           (p : GPT-Params Γ vo ed sl fd)
           (x : E Γ (ar $ (bs ⊗ sl) ⊗ ed))
           → E Γ (ar $ (bs ⊗ sl) ⊗ ed)
    mlp {ed = ed} {sl = sl} {fd = fd} {bs = bs} p x =
      m-linear {ed} ⟨ p .wdown ⟩ $
        relu $ m-linear {fd} ⟨ p .wup ⟩ ⟨ x ⟩

    block : ∀ {Γ}
            (pf : ah * hd ≈ ed) (sc : ℕ)
            (p : GPT-Params Γ vo ed sl fd)
            (mask : E Γ (ar $ bs ⊗ (sl ⊗ sl)))
            (h : E Γ (ar $ (bs ⊗ sl) ⊗ ed))
            → E Γ (ar $ (bs ⊗ sl) ⊗ ed)
    block {ah} {hd} {ed} {vo} {sl} {fd} {bs} pf sc p mask x =
      Let a := attn pf sc p mask (m-rmsnorm {bs ⊗ sl} x) In
      Let x+a := ⟨ x ⟩ ⊞ a In
      Let m := mlp ⟨ p ⟩ₚ (m-rmsnorm {bs ⊗ sl} x+a)
      In x+a ⊞ m

    model : ∀ {Γ}
            (pf : ah * hd ≈ ed) (sc : ℕ)
            (p : GPT-Params Γ vo ed sl fd)
            (mask : E Γ (ar $ bs ⊗ (sl ⊗ sl)))
            (te : E Γ (ar $ (bs ⊗ sl) ⊗ ed))
            → E Γ (ar $ (bs ⊗ sl) ⊗ vo)
    model {vo = vo} {sl = sl} {bs = bs} pf sc p mask te =
      -- DANGER : iassᵣ is inneficient. Opt should be able to take care of it
      -- NOTE : microgpt applies normalization twice for some reason
      Let h := (m-rmsnorm {bs ⊗ sl} $ te ⊞ iassᵣ {bs} (tile {bs} (p .wpe))) In
      -- TODO : Generalize to multiple layers
      -- DANGER : microgpt does NOT normalize after block but gpt2 does?
      Let h₁ := block pf sc ⟨ p ⟩ₚ ⟨ mask ⟩ h In
        m-linear {vo} ⟨ p .wvoc ⟩ h₁

    cross-entropy : ∀ {Γ} (logits target : E Γ (ar s)) → (E Γ (ar []))
    cross-entropy {s} logits target =
      Let lnsf := ln (ℙ logits) In
      (⊟ (Sum λ i → sels lnsf i ⊠ sels ⟨ target ⟩ i))

    -- cross-entropy : ∀ {Γ} (logits target : E Γ (ar s)) → (E Γ (ar []))
    -- cross-entropy {s} logits target =
    --   (⊟ (Sum λ i → sels (ln (ℙ ⟨ logits ⟩)) i ⊠ sels ⟨ target ⟩ i))

    m-cross-entropy : ∀ {Γ} (logits target : E Γ (ar (s ⊗ p))) → (E Γ (ar s))
    m-cross-entropy {s} {p} logits target =
      Imaps λ i → cross-entropy {p} (sel ⟨ logits ⟩ i) (sel ⟨ target ⟩ i)

    avg : ∀ {Γ} → E Γ (ar s) → E Γ (ar [])
    avg {s} x = scaledown (len s) (Sum λ i → sels ⟨ x ⟩ i)

    gpt-loss : ∀ {Γ}
            (pf : ah * hd ≈ ed) (sc : ℕ)
            (p : GPT-Params Γ vo ed sl fd)
            (mask : E Γ (ar $ bs ⊗ (sl ⊗ sl)))
            (te : E Γ (ar $ (bs ⊗ sl) ⊗ ed))
            (target : E Γ (ar $ (bs ⊗ sl) ⊗ vo))
            → E Γ (ar [])
    gpt-loss {vo = vo} {sl = sl} {bs = bs} pf sc p mask te target =
      Let logits := model pf sc p mask te In
      Let losses := m-cross-entropy {bs ⊗ sl} logits ⟨ target ⟩ In
      -- DANGER: Not sure how to deal with batchsize
      avg losses

    gpt-loss-flat : ∀ {Γ}
            (pf : ah * hd ≈ ed) (sc : ℕ)
            (p : GPT-Params Γ vo ed sl fd)
            (mask : E Γ (ar $ bs ⊗ (sl ⊗ sl)))
            (te : E Γ (ar $ (bs ⊗ sl) ⊗ ed))
            (target : E Γ (ar $ (bs ⊗ sl) ⊗ vo))
            → E Γ (ar [])
    gpt-loss-flat {ah = ah} {ed = ed} {vo = vo} {sl = sl} {fd = fd} {bs = bs}
      pf sc p mask te target =
      -- DANGER : iassᵣ is inneficient. Opt should be able to take care of it
      Let e := (te ⊞ iassᵣ {bs} (tile {bs} (p .wpe))) In
      -- NOTE : microgpt applies normalization twice for some reason.
        -- Probably an effect of having only one layer
      -- first rmsnorm
      Let ee₀ := e ⊠ e In
      Let ms₀ := scaledown (len ed) (Imaps {bs ⊗ sl} λ i → Sum λ j →
        sels (sel ee₀ i) j) In
      Let scale₀ := √ (ms₀ ⊞ (scaledown 100000 𝟙)) In
      Let n₀ :=
        (Imap λ i → Imaps λ j → sels (sel e i) j // sels scale₀ i) In
      -- second rmsnorm
      Let ee₁ := n₀ ⊠ n₀ In
      Let ms₁ := scaledown (len ed) (Imaps {bs ⊗ sl} λ i → Sum λ j →
        sels (sel ee₁ i) j) In
      Let scale₁ := √ (ms₁ ⊞ (scaledown 100000 𝟙)) In
      Let n₁ :=
        (Imap λ i → Imaps λ j → sels (sel n₀ i) j // sels scale₁ i) In
      -- attn
      Let q₀ := m-linear {ed} ⟨ p .wqry ⟩ n₁ In
      Let k₀ := m-linear {ed} ⟨ p .wkey ⟩ n₁ In
      Let v₀ := m-linear {ed} ⟨ p .wval ⟩ n₁ In
      Let q₁ := split-heads {bs} pf q₀ In
      Let k₁ := split-heads {bs} pf k₀ In
      Let v₁ := split-heads {bs} pf v₀ In
      Let q := imedial {bs} q₁ In
      Let k := imedial {bs} k₁ In
      Let v := imedial {bs} v₁ In
      Let w := m-matmult {bs ⊗ ah} {sl} q k In
      Let masked := mask-attn-weights {bs} {sl} ⟨ mask ⟩ (scaledown sc w) In
      Let sf := gpt-softmax {bs} {ah} {sl} masked In
      Let a₀ := m-matmul {bs ⊗ ah} {sl} sf v In
      -- Let a₁ := merge-heads {bs} pf a₀ In
      Let a₁ := imedial {bs} {ah} {sl} a₀ In
      Let a₂ := merge-heads {bs} pf a₁ In
      Let a := m-linear {ed} ⟨ p .wout ⟩ a₂ In
      Let x+a := n₁ ⊞ a In
      -- third rmsnorm
      Let ee₂ := x+a ⊠ x+a In
      Let ms₂ := scaledown (len ed) (Imaps {bs ⊗ sl} λ i → Sum λ j →
        sels (sel ee₂ i) j) In
      Let scale₂ := √ (ms₂ ⊞ (scaledown 100000 𝟙)) In
      Let n₂ :=
        (Imap λ i → Imaps λ j → sels (sel x+a i) j // sels scale₂ i) In
      -- mlp
      Let m₀ := m-linear {fd} ⟨ p .wup ⟩ n₂ In
      Let re := relu m₀ In
      Let m := m-linear {ed} ⟨ p .wdown ⟩ re In
      Let x+a+m := x+a ⊞ m In
      Let logits := m-linear {vo} ⟨ p .wvoc ⟩ x+a+m In
      -- cross entropy
      Let losses := m-cross-entropy {bs ⊗ sl} logits ⟨ target ⟩
      In avg losses

    BS = ι 1 ; SL = ι 16
    ED = ι 16 ; AH = ι 4 ; HD = ι 4
    FD = ι 64 ; SC = 2 ; VO = ι 27

    PF : AH * HD ≈ ED
    PF = cons

    model-e : E _ _
    model-e = Lcon (_ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _) _ ε
      λ wpe wqry wkey wval wout wup wdown wvoc mask te →
        model {vo = VO} {sl = SL} {fd = FD} {bs = BS} PF SC
        (to-gptp wpe wqry wkey wval wout wup wdown wvoc) mask te

    gpt-loss-e : E _ _
    gpt-loss-e = Lcon (_ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _) _ ε
      λ wpe wqry wkey wval wout wup wdown wvoc mask te target →
        gpt-loss {vo = VO} {sl = SL} {fd = FD} {bs = BS} PF SC
        (to-gptp wpe wqry wkey wval wout wup wdown wvoc) mask te target

    gpt-loss-flat-e : E _ _
    gpt-loss-flat-e = Lcon (_ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _ ∷ _) _ ε
      λ wpe wqry wkey wval wout wup wdown wvoc mask te target →
        gpt-loss-flat {vo = VO} {sl = SL} {fd = FD} {bs = BS} PF SC
        (to-gptp wpe wqry wkey wval wout wup wdown wvoc) mask te target