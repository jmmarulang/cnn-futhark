{-# OPTIONS  --backtracking-instance-search #-}
-- {-# OPTIONS --warn=noUserWarning #-}
module _ where
module _ where
  open import Data.Nat using (ℕ; zero; suc)
  open import Data.List as L using (List; []; _∷_)
  open import Ar hiding (sum; slide; backslide; imapb; selb)
  open import Relation.Binary.PropositionalEquality
  open import Relation.Nullary
  infixl 15 _▹_

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
  infixl 15 _⊔_

  data Bop : Set where
    plus mul : Bop

  -- Jairo made
  data Uop : Set where
    logistic neg
    -- Jairo made
      exp rectifier squared inverse ind-positive logarithm -- What happends with undefined terms like squared -2?
      : Uop

  unit : S
  unit = []

  data E : Ctx → IS → Set where
    var        : is ∈ Γ → E Γ is
    zero       : E Γ (ar s)
    one        : E Γ (ar s)

    imaps      : E (Γ ▹ ix s) (ar unit) → E Γ (ar s)
    sels       : E Γ (ar s) → E Γ (ix s) → E Γ (ar unit)

    imap       : E (Γ ▹ ix s) (ar p) → E Γ (ar (s ⊗ p))
    sel        : E Γ (ar (s ⊗ p)) → E Γ (ix s) → E Γ (ar p)

    imapb      : s * p ≈ q → E (Γ ▹ ix s) (ar p) → E Γ (ar q)
    selb       : s * p ≈ q → E Γ (ar q) → E Γ (ix s) → E Γ (ar p)

    sum        : E (Γ ▹ ix s) (ar p) → E Γ (ar p)
    zero-but   : E Γ (ix s) → E Γ (ix s) → E Γ (ar p) → E Γ (ar p)

    slide      : E Γ (ix s) → s + p ≈ r → E Γ (ar r) → suc p ≈ u → E Γ (ar u)
    backslide  : E Γ (ix s) → E Γ (ar u) → suc p ≈ u → s + p ≈ r → E Γ (ar r)

    -- logi   : E Γ (ar s) → E Γ (ar s)
    bin        : Bop → E Γ (ar s) → E Γ (ar s) → E Γ (ar s)
    scaledown  : ℕ → E Γ (ar s) → E Γ (ar s)
    -- ⊟_      : E Γ (ar s) → E Γ (ar s)
    let′       : E Γ (ar s) → E (Γ ▹ ar s) (ar p) → E Γ (ar p)
    -- Jairo made
    un : Uop → E Γ (ar s) → E Γ (ar s)

  pattern 𝟙 = one
  pattern 𝟘 = zero

  pattern ⊟_ a = un neg a
  pattern 𝕖^_ a = un exp a
  pattern logi a = un logistic a
  pattern sqrt a = un squared a
  pattern 𝟙/ a = un inverse a
  pattern relu a = un rectifier a
  pattern ln a = un logarithm a
  pattern 𝕀+ a = un ind-positive a

  pattern _⊞_ a b = bin plus a b
  pattern _⊠_ a b = bin mul a b

  -- infixl 5 𝕀+
  -- syntax 𝕀+ a b = 𝕀[ a < b ]

  _⊟_ : ( a b : E Γ (ar s)) → E Γ (ar s)
  _⊟_ a b = a ⊞ ⊟ b

  -- maximum
  _⊔_ : ( a b : E Γ (ar s)) → E Γ (ar s)
  a ⊔ b = a ⊞ (relu (b ⊟ a))

  _//_ : ( a b : E Γ (ar s)) → E Γ (ar s)
  _//_ a b = a ⊠ (𝟙/ b)

  𝟚 : E Γ (ar s)
  𝟚 = 𝟙 ⊞ 𝟙

module WkSub where
  open import Data.Nat using (ℕ; zero; suc; _+_)
  open import Relation.Binary.PropositionalEquality
  open import Function

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
  wk s zero = zero
  wk s one = one
  wk s (imaps e) = imaps (wk (keep s) e)
  wk s (sels e e₁) = sels (wk s e) (wk s e₁)
  wk s (imap e) = imap (wk (keep s) e)
  wk s (sel e e₁) = sel (wk s e) (wk s e₁)
  wk s (imapb x e) = imapb x (wk (keep s) e)
  wk s (selb x e e₁) = selb x (wk s e) (wk s e₁)
  wk s (sum e) = sum (wk (keep s) e)
  wk s (zero-but e e₁ e₂) = zero-but (wk s e) (wk s e₁) (wk s e₂)
  wk s (slide e x e₁ x₁) = slide (wk s e) x (wk s e₁) x₁
  wk s (backslide e e₁ x x₁) = backslide (wk s e) (wk s e₁) x x₁
  -- wk s (logi e) = logi (wk s e)
  wk s (bin x e e₁) = bin x (wk s e) (wk s e₁)
  wk s (scaledown x e) = scaledown x (wk s e)
  -- wk s (⊟_ e) = ⊟_ (wk s e)
  wk s (let′ e e₁) = let′ (wk s e) (wk (keep s) e₁)
  -- Jairo made
  wk s (un x e) = un x (wk s e)

  _∙ʷ_ : Δ ⊆ Ψ → Γ ⊆ Δ → Γ ⊆ Ψ
  s ∙ʷ ε = s
  skip s ∙ʷ skip p = skip (s ∙ʷ skip p)
  keep s ∙ʷ skip p = skip s ∙ʷ p
  skip s ∙ʷ keep p = skip (s ∙ʷ keep p)
  keep s ∙ʷ keep p = keep (s ∙ʷ p)

  ⊆-eq : Γ ⊆ Γ
  ⊆-eq {ε} = ε
  ⊆-eq {Γ ▹ x} = keep ⊆-eq

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
  sub zero s = zero
  sub one s = one
  sub (imaps e) s = imaps (sub e (skeep s))
  sub (sels e e₁) s = sels (sub e s) (sub e₁ s)
  sub (imap e) s = imap (sub e (skeep s))
  sub (sel e e₁) s = sel (sub e s) (sub e₁ s)
  sub (imapb x e) s = imapb x (sub e (skeep s))
  sub (selb x e e₁) s = selb x (sub e s) (sub e₁ s)
  sub (sum e) s = sum (sub e (skeep s))
  sub (zero-but e e₁ e₂) s = zero-but (sub e s) (sub e₁ s) (sub e₂ s)
  sub (slide e x e₁ x₁) s = slide (sub e s) x (sub e₁ s) x₁
  sub (backslide e e₁ x x₁) s = backslide (sub e s) (sub e₁ s) x x₁
  -- sub (logi e) s = logi (sub e s)
  sub (bin x e e₁) s = bin x (sub e s) (sub e₁ s)
  sub (scaledown x e) s = scaledown x (sub e s)
  -- sub (⊟_ e) s = ⊟_ (sub e s)
  sub (let′ e e₁) s = let′ (sub e s) (sub e₁ (skeep s))
  -- Jairo made
  sub (un x e) s = un x (sub e s)

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
  sub-at-id zero = refl
  sub-at-id one = refl
  sub-at-id (imaps e) = cong imaps (sub-at-id e)
  sub-at-id (sels e e₁) = cong₂ sels (sub-at-id e) (sub-at-id e₁)
  sub-at-id (imap e) = cong imap (sub-at-id e)
  sub-at-id (sel e e₁) = cong₂ sel (sub-at-id e) (sub-at-id e₁)
  sub-at-id (imapb x e) = cong (imapb x) (sub-at-id e)
  sub-at-id (selb x e e₁) = cong₂ (selb x) (sub-at-id e) (sub-at-id e₁)
  sub-at-id (sum e) = cong sum (sub-at-id e)
  sub-at-id (zero-but e e₁ e₂) rewrite (sub-at-id e) | sub-at-id e₁ | sub-at-id e₂ = refl
  sub-at-id (slide e x e₁ x₁) rewrite sub-at-id e | sub-at-id e₁ = refl
  sub-at-id (backslide e e₁ x x₁) rewrite sub-at-id e | sub-at-id e₁ = refl
  -- sub-at-id (logi e) = cong logi (sub-at-id e)
  sub-at-id (bin x e e₁) = cong₂ (bin x) (sub-at-id e) (sub-at-id e₁)
  sub-at-id (scaledown x e) = cong (scaledown x) (sub-at-id e)
  -- sub-at-id (⊟_ e) = cong ⊟_ (sub-at-id e)
  sub-at-id (let′ e e₁) = cong₂ let′ (sub-at-id e) (sub-at-id e₁)
  -- Jairo made
  sub-at-id (un x e) = cong (un x) (sub-at-id e)

  sub-ε : (e : E ε is) → sub e ε ≡ e
  sub-ε e = sub-at-id e

  sub-swap : Sub (Γ ▹ is ▹ ip) (Γ ▹ ip ▹ is)
  sub-swap = (sdrop (sdrop sub-id) ▹ var v₀) ▹ var (there v₀)

  -- We are not really using this, but this is a useful function to have.
  open import Data.Maybe
  strenv : (x : is ∈ Γ) (y : ip ∈ Γ) → Maybe (ip ∈ (Γ / x))
  strenv v₀ v₀ = nothing
  strenv v₀ (there y) = just y
  strenv (there x) v₀ = just v₀
  strenv (there x) (there y) = map there (strenv x y)

  stren : E Γ is → (v : ip ∈ Γ) → Maybe (E (Γ / v) is)
  stren (var x) v = map var (strenv v x)
  stren zero v = just zero
  stren one v = just one
  stren (imaps e) v = map imaps (stren e (there v))
  stren (sels e e₁) v = do
    l ← stren e v
    r ← stren e₁ v
    just (sels l r)
  stren (imap e) v = map imap (stren e (there v))
  stren (sel e e₁) v = do
    l ← stren e v
    r ← stren e₁ v
    just (sel l r)
  stren (imapb x e) v = map (imapb x) (stren e (there v))
  stren (selb x e e₁) v = do
    l ← stren e v
    r ← stren e₁ v
    just (selb x l r)
  stren (sum e) v = map sum (stren e (there v))
  stren (zero-but e e₁ e₂) v = do
    a ← stren e v
    b ← stren e₁ v
    c ← stren e₂ v
    just (zero-but a b c)
  stren (slide e x e₁ x₁) v = do
    a ← stren e v
    b ← stren e₁ v
    just (slide a x b x₁)
  stren (backslide e e₁ x x₁) v = do
    a ← stren e v
    b ← stren e₁ v
    just (backslide a b x x₁)
  -- stren (logi e) v = map logi (stren e v)
  stren (bin x e e₁) v = do
    a ← stren e v
    b ← stren e₁ v
    just (bin x a b)
  stren (scaledown x e) v = map (scaledown x) (stren e v)
  stren (⊟ e) v = map ⊟_ (stren e v)
  stren (let′ e e₁) v = do
    a ← stren e v
    b ← stren e₁ (there v)
    just (let′ a b)
  -- Jairo made
  stren (un x e) v = map (un x) (stren e v)

  -- Get rid of lets that do not use their arguments.
  norm-lets : E Γ is → E Γ is
  norm-lets (var x) = (var x)
  norm-lets zero = zero
  norm-lets one = one
  norm-lets (imaps e) = imaps (norm-lets e)
  norm-lets (sels e e₁) = sels (norm-lets e) (norm-lets e₁)
  norm-lets (imap e) = imap (norm-lets e)
  norm-lets (sel e e₁) = sel (norm-lets e) (norm-lets e₁)
  norm-lets (imapb x e) = imapb x (norm-lets e)
  norm-lets (selb x e e₁) = selb x (norm-lets e) (norm-lets e₁)
  norm-lets (sum e) = sum (norm-lets e)
  norm-lets (zero-but e e₁ e₂) = zero-but (norm-lets e) (norm-lets e₁) (norm-lets e₂)
  norm-lets (slide e x e₁ x₁) = slide (norm-lets e) x (norm-lets e₁) x₁
  norm-lets (backslide e e₁ x x₁) = backslide (norm-lets e) (norm-lets e₁) x x₁
  -- norm-lets (logi e) = logi (norm-lets e)
  norm-lets (bin x e e₁) = bin x (norm-lets e) (norm-lets e₁)
  norm-lets (scaledown x e) = scaledown x (norm-lets e)
  -- norm-lets (⊟_ e) = ⊟_ (norm-lets e)
  norm-lets (let′ e e₁) = maybe id (let′ (norm-lets e) (norm-lets e₁)) (stren (norm-lets e₁) v₀)
  -- Jairo made
  norm-lets (un x e) = un x (norm-lets e)

  count-uses : E Γ is → ip ∈ Γ → ℕ
  count-uses (var x) v with eq? x v
  ... | veq = 1
  ... | _ = 0
  count-uses zero v = 0
  count-uses one v = 0
  count-uses (imaps e) v = count-uses e (there v)
  count-uses (sels e e₁) v = count-uses e v + count-uses e₁ v
  count-uses (imap e) v = count-uses e (there v)
  count-uses (sel e e₁) v = count-uses e v + count-uses e₁ v
  count-uses (imapb x e) v = count-uses e (there v)
  count-uses (selb x e e₁) v = count-uses e v + count-uses e₁ v
  count-uses (sum e) v = count-uses e (there v)
  count-uses (zero-but e e₁ e₂) v = count-uses e v + count-uses e₁ v + count-uses e₂ v
  count-uses (slide e x e₁ x₁) v = count-uses e v + count-uses e₁ v
  count-uses (backslide e e₁ x x₁) v = count-uses e v + count-uses e₁ v
  -- count-uses (logi e) v = count-uses e v
  count-uses (bin x e e₁) v = count-uses e v + count-uses e₁ v
  count-uses (scaledown x e) v = count-uses e v
  -- count-uses (⊟_ e) v = count-uses e v
  count-uses (let′ e e₁) v = count-uses e v + count-uses e₁ (there v)
  -- Jairo made
  count-uses (un x e) v = count-uses e v

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

  module Cnn where

    conv : ∀ {Γ} → E Γ (ar r) → ⦃ s + p ≈ r ⦄ → E Γ (ar s) → ⦃ suc p ≈ u ⦄
        → E Γ (ar u)
    conv f ⦃ s+p ⦄ g ⦃ ss ⦄
      = Sum λ i → (slide i s+p ⟨ f ⟩ ss) ⊠ Imaps λ j → sels ⟨ g ⟩ i

    mconv : ⦃ s + p ≈ r ⦄ → (inp : E Γ (ar r)) (ws : E Γ (ar (u ⊗ s)))
            (bᵥ : E Γ (ar u)) → ⦃ suc p ≈ w ⦄ → E Γ (ar (u ⊗ w))
    mconv ⦃ sp ⦄ inp wᵥ bᵥ ⦃ su ⦄ =
      Imap λ i → conv ⟨ inp ⟩ (sel ⟨ wᵥ ⟩ i) ⊞ Imaps λ _ → sels ⟨ bᵥ ⟩ i

    avgp₂ : ∀ m n → (a : E Γ (ar (m ℕ.* 2 ∷ n ℕ.* 2 ∷ [])))
          → E Γ (ar (m ∷ n ∷ []))
    avgp₂ m n a =
      Imaps λ i → scaledown 4 $ Sum λ j → sels (selb it ⟨ a ⟩ i) j

    sqerr : (r o : E Γ (ar [])) → E Γ (ar [])
    sqerr r o = scaledown 2 ((r ⊞ (⊟_ o)) ⊠ (r ⊞ (⊟_ o)))

    meansqerr : (r o : E Γ (ar s)) → E Γ (ar [])
    meansqerr r o = Sum λ i → sqerr (sels ⟨ r ⟩ i) (sels ⟨ o ⟩ i)

    cnn : E _ _
    cnn = Lcon (  ar (28 ∷ 28 ∷ []) ∷ ar (6 ∷ 5 ∷ 5 ∷ [])
                ∷ ar (6 ∷ [])       ∷ ar (12 ∷ 6 ∷ 5 ∷ 5 ∷ [])
                ∷ ar (12 ∷ [])      ∷ ar (10 ∷ 12 ∷ 1 ∷ 4 ∷ 4 ∷ [])
                ∷ ar (10 ∷ [])      ∷ ar (10 ∷ 1 ∷ 1 ∷ 1 ∷ 1 ∷ [])
                -- ∷ ar (10 ∷ 1 ∷ 1 ∷ 1 ∷ 1 ∷ [])
                ∷ [])
              -- (ar (10 ∷ 1 ∷ 1 ∷ 1 ∷ 1 ∷ [])) ε
              (ar ([])) ε
          λ inp k₁ b₁ k₂ b₂ fc b target →
          Let c₁₁ := mconv inp k₁ b₁  In
          Let c₁  := logi c₁₁ In
          Let s₁  := (Imap {s = 6 ∷ []} λ i → avgp₂ 12 12 (sel c₁ i)) In
          Let c₂₁ := mconv s₁ k₂ b₂ In
          Let c₂  := logi c₂₁ In
          Let s₂  := (Imap {s = 12 ∷ 1 ∷ []} λ i → avgp₂ 4 4 (sel c₂ i)) In
          Let o₁  := mconv s₂ fc b In
          Let o   := logi o₁ In
          -- Mean squared error
          Let e   := meansqerr target o In
          e

  module Microgpt where

    open import Data.Product as Prod hiding (_<*>_)
    open import Data.List.Properties

    variable
      ah hd ed sl vo pr fd : S

    infixr 6 _⊡_
    _⊡_ = trans

    -- can i get rid of substitutions?
    -- subst-shape : ∀ {s p} → s ≡ p → E Γ (ar s) → E Γ (ar p)
    -- subst-shape refl b = b

    -- subst-idrl : ∀ {s} → E Γ (ar (s ⊗ [])) → E Γ (ar s)
    -- subst-idrl x = subst-shape (++-identityʳ _) x

    -- subst-idrr : ∀ {s} → E Γ (ar s) → E Γ (ar (s ⊗ []))
    -- subst-idrr {Γ} {s} x = subst-shape (sym (++-identityʳ _)) x

    -- subst-idll : ∀ {s} → E Γ (ar ([] ⊗ s)) → E Γ (ar s)
    -- subst-idll x = subst-shape (++-identityˡ _) x

    -- subst-assl : ∀ {Γ} → E Γ (ar ((s ⊗ p) ⊗ q)) → E Γ (ar (s ⊗ (p ⊗ q)))
    -- subst-assl {s} {p} {q} {Γ} x = subst-shape (++-assoc s p q) x

    -- subst-assr : ∀ {Γ} → E Γ (ar (s ⊗ (p ⊗ q))) → E Γ (ar ((s ⊗ p) ⊗ q))
    -- subst-assr {s} {p} {q} {Γ} x = subst-shape (sym (++-assoc s p q)) x

    pw3-subst : ∀ {R} {s1 s2 p1 p2 q1 q2 : S}
                → (s1 ≡ s2) → (p1 ≡ p2) → (q1 ≡ q2)
                → (Pointw₃ R s1 p1 q1) → Pointw₃ R s2 p2 q2
    pw3-subst refl refl refl pw = pw

    pw3-con : ∀ {s1 s2 p1 p2 q1 q2 : S} {R} → Pointw₃ R s1 p1 q1
         → Pointw₃ R s2 p2 q2
         → Pointw₃ R (s1 ⊗ s2) (p1 ⊗ p2) (q1 ⊗ q2)
    pw3-con [] [] = []
    pw3-con [] cons = cons
    pw3-con cons [] =
      pw3-subst (sym ++-neutʳ) (sym ++-neutʳ) (sym ++-neutʳ) cons
    pw3-con {s1 ∷ s1s} {s2 ∷ s2s} {p1 ∷ p1s} {p2 ∷ p2s} {q1 ∷ q1s} {q2 ∷ q2s}
      (cons ⦃ a1 ⦄ ⦃ b1 ⦄) (cons ⦃ a2 ⦄ ⦃ b2 ⦄) = cons ⦃ a1 ⦄ ⦃ g ⦄ where
        g = pw3-con b1 (cons ⦃ a2 ⦄ ⦃ b2 ⦄)

    pw3-dup : ∀ {R} {s p q : S} → Pointw₃ R s p q
            → Pointw₃ R (s ⊗ s) (p ⊗ p) (q ⊗ q)
    pw3-dup pw = pw3-con pw pw

    icom : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (s ⊗ u))
    icom {u} {s} x = Imap {s} λ i → Imaps λ j → sels (sel ⟨ x ⟩ j) i

    iswap3 : ∀ {Γ} → E Γ (ar (p ⊗ (s ⊗ u))) → E Γ (ar (s ⊗ (p ⊗ u)))
    iswap3 {p} {s} {u} x = Imap {s} λ i → Imap {p} λ j → sel (sel ⟨ x ⟩ j) i

    iass-r : ∀ {Γ} → E Γ (ar ((p ⊗ s) ⊗ u)) → E Γ (ar (p ⊗ (s ⊗ u)))
    iass-r {p} {s} {u} {Γ} x = Imap {p} λ i → Imap {s} λ j → Imaps λ z → 
      sels (sel (sel (icom {s = u} ⟨ x ⟩) z) i) j

    iass-l : ∀ {Γ} → E Γ (ar (p ⊗ (s ⊗ u))) → E Γ (ar ((p ⊗ s) ⊗ u))
    iass-l {p} {s} {u} {Γ} x = 
      icom {u = u} $ Imap {u} λ i → Imap {p} λ j → Imaps λ z → 
        sels (sel (sel ⟨ x ⟩ j) z) i
    
    iswap4in : ∀ {Γ} → E Γ (ar ((s ⊗ p) ⊗ (q ⊗ r)))
              → E Γ (ar ((s ⊗ q) ⊗ (p ⊗ r)))
    iswap4in {s} {p} {q} {r} {Γ} x = 
      iass-l {s} $ Imap {s} λ i → iswap3 {p} {q} (sel (iass-r {s} ⟨ x ⟩) i)

    linear : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar s) → E Γ (ar u)
    linear {u} {s} w x =
      Imaps {u} λ i → Sum {s} λ j → sels (sel ⟨ w ⟩ i) j ⊠ sels ⟨ x ⟩ j

    matmul : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (s ⊗ r)) → E Γ (ar (u ⊗ r))
    matmul {u} {s} {r} w x = Imap {u} λ i →
      Imaps (λ j → sels (linear ⟨ w ⟩ (Imaps λ k → sels (sel ⟨ x ⟩ k) j) ) i)

    m-linear : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (p ⊗ s)) → E Γ (ar (p ⊗ u))
    m-linear {u} {s} {p} w xs = Imap {p} λ i → linear ⟨ w ⟩ (sel ⟨ xs ⟩ i)

    -- Is this correct?
    softmax : ∀ {Γ} → E Γ (ar s) → E Γ (ar s)
    softmax {s = s} x =
      Let exps := 𝕖^ (x) In -- subst one for stability
      Let total := Sum {s} (λ i → sels exps i) In 
      Let r := Imaps {s} (λ i → (sels exps i) // total) In r

    m-softmax : ∀ {Γ} → E Γ (ar (s ⊗ p)) → E Γ (ar (s ⊗ p))
    m-softmax {s} {p} {Γ} x = Imap {s} λ i → softmax (sel ⟨ x ⟩ i)

    -- add a small number to avoid dividing by zero?
    rmsnorm : ∀ {Γ} → E Γ (ar s) → E Γ (ar s)
    rmsnorm {s = s} x =
      Let ms := scaledown (len s) (Sum (λ i → sels ⟨ x ⟩ i ⊠ sels ⟨ x ⟩ i)) In -- always positive
      Let scale := 𝟙/ (sqrt (ms ⊞ scaledown 100000 one)) In -- add a small number to avoid dividinx by zero
      Let r := Imaps (λ i → sels ⟨ x ⟩ i ⊠ scale) In r

    m-rmsnorm : ∀ {Γ} → E Γ (ar (s ⊗ p)) → E Γ (ar (s ⊗ p))
    m-rmsnorm {s} {p} {Γ} x = Imap {s} λ i → rmsnorm (sel ⟨ x ⟩ i)

    -- max : ∀ {Γ} → E Γ (ar s) → E Γ (ar s) → E Γ (ar s)
    -- max x y = x ⊞ relu (y ⊟ x)

    avg : ∀ {Γ} → E Γ (ar s) → E Γ (ar [])
    avg {s} x = scaledown (len s) (Sum λ i → sels ⟨ x ⟩ i)

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

    wkgptp : ∀ {Γ Δ} → Prefix Γ Δ → GPT-Params Γ vo ed sl fd
            → GPT-Params Δ vo ed sl fd
    wkgptp pre p .wpe = wkp pre (wpe p)
    wkgptp pre p .wqry = wkp pre (wqry p)
    wkgptp pre p .wkey = wkp pre (wkey p)
    wkgptp pre p .wval = wkp pre (wval p)
    wkgptp pre p .wout = wkp pre (wout p)
    wkgptp pre p .wup = wkp pre (wup p)
    wkgptp pre p .wdown = wkp pre (wdown p)
    wkgptp pre p .wvoc = wkp pre (wvoc p)

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

    attention : ∀ {Γ} (sc : ℕ) (mask : E Γ (ar (sl ⊗ sl)))
                (qs ks vs : E Γ (ar (sl ⊗ hd))) → E Γ (ar (sl ⊗ hd))
    attention {sl} {hd} sc mask qs ks vs = 
      Let qks := matmul {u = sl} qs (icom {sl} ks) In 
      Let scqks := scaledown sc qks In 
      Let masked := scqks ⊞ ⟨ mask ⟩ In
      Let sf := m-softmax {sl} masked In 
      Let r := matmul {r = hd} sf ⟨ vs ⟩ In r
      -- matmul {r = hd}
      -- (m-softmax {sl} (
      --   scaledown sc ⟨ matmul {u = sl} qs (icom {sl} ks) ⟩ ⊞ mask)) vs

    mh-attention : let ed = ah ⊗ hd in ∀ {Γ} (sc : ℕ)
                   (mask : E Γ (ar (sl ⊗ sl)))
                   (qs ks vs : E Γ (ar (sl ⊗ ed)))
                  → E Γ (ar (sl ⊗ ed))
    mh-attention {ah} {hd} {sl} {Γ} sc mask qs ks vs =
        iswap3 {ah} {sl} $ Imap {ah} λ i → attention {sl} sc ⟨ mask ⟩
          (sel (iswap3 {sl} {ah} ⟨ qs ⟩) i)
          (sel (iswap3 {sl} {ah} ⟨ ks ⟩) i)
          (sel (iswap3 {sl} {ah} ⟨ vs ⟩) i)

    mlp : ∀ {Γ} (wup : E Γ (ar (fd ⊗ ed)))
          (wdown : E Γ (ar (ed ⊗ fd))) (seq : E Γ (ar (sl ⊗ ed)))
          → E Γ (ar (sl ⊗ ed))
    mlp {fd} {ed} {sl} wup wdown seq =
      Let nseq := m-rmsnorm {sl} seq In
      Let useq := m-linear {p = sl} ⟨ wup ⟩ nseq In
      Let aseq := relu useq In
      Let dseq := m-linear {u = ed} {p = sl} ⟨ wdown ⟩ aseq In
      Let cseq := dseq ⊞ ⟨ seq ⟩ In cseq

    mgpt-layer : let ed = ah ⊗ hd in ∀ {Γ} (sc : ℕ) (mask : E Γ (ar (sl ⊗ sl)))
                 (p : GPT-Params Γ vo ed sl fd) (seq : E Γ (ar (sl ⊗ ed)))
                 → E Γ (ar (sl ⊗ ed))
    mgpt-layer {ah} {hd} {sl} {vo} {fd} {Γ} sc mask p seq = let ed = ah ⊗ hd in
      Let nseq := m-rmsnorm {sl} seq In
      Let qs := m-linear {u = ed} {p = sl} ⟨ p .wqry ⟩ nseq In
      Let ks := m-linear {u = ed} {p = sl} ⟨ p .wkey ⟩ nseq In
      Let vs := m-linear {u = ed} {p = sl} ⟨ p .wval ⟩ nseq In
      Let attn := mh-attention {hd = hd} {sl = sl} sc ⟨ mask ⟩ qs ks vs In
      Let oseq := m-linear {u = ed} {p = sl} ⟨ p .wout ⟩ attn In
      Let cseq := oseq ⊞ ⟨ seq ⟩ In
      Let fseq := mlp {sl = sl} ⟨ p .wup ⟩ ⟨ p .wdown ⟩ cseq In fseq

    -- make sure the order makes sense
    block-tok : ∀ {Γ} → E Γ (ar ed) → ah * hd ≈ ed → E Γ (ar (ah ⊗ hd))
    block-tok {ed} {ah} {hd} {Γ} x pr = Imap {ah} λ i → selb pr ⟨ x ⟩ i

    block-seq : ∀ {Γ} → E Γ (ar (sl ⊗ ed)) → ah * hd ≈ ed
                → E Γ (ar (sl ⊗ (ah ⊗ hd)))
    block-seq {sl} {ed} {ah} x pr = Imap {sl} λ i → block-tok (sel ⟨ x ⟩ i) pr

    block-w : ∀ {Γ} → E Γ (ar (ed ⊗ ed)) → ah * hd ≈ ed
                → E Γ (ar ((ah ⊗ hd) ⊗ (ah ⊗ hd)))
    block-w {ed} {ah} {hd} {Γ} x pr = iswap4in {ah} {ah} $
      Imap {ah ⊗ ah} λ i → selb (pw3-dup pr) ⟨ x ⟩ i

    block-param : ∀ {Γ} (p : GPT-Params Γ vo ed sl fd)
                   → ah * hd ≈ ed → GPT-Params Γ vo (ah ⊗ hd) sl fd
    block-param p pr .wpe = block-seq (wpe p) pr
    block-param p pr .wqry = block-w (wqry p) pr
    block-param p pr .wkey = block-w (wkey p) pr
    block-param p pr .wval = block-w (wval p) pr
    block-param p pr .wout = block-w (wout p) pr
    block-param p pr .wup = block-seq (wup p) pr
    block-param {ed = ed} {fd = fd} p pr .wdown =
      icom {fd} (block-seq (icom {ed} (wdown p)) pr)
    block-param p pr .wvoc = block-seq (wvoc p) pr

    mgpt-forward : ∀ {ah hd : S} {Γ} (sc : ℕ) (mask : E Γ (ar (sl ⊗ sl)))
                   (p : GPT-Params Γ vo ed sl fd) (wseq : E Γ (ar (sl ⊗ ed)))
                   → ah * hd ≈ ed → E Γ (ar (sl ⊗ vo))
    mgpt-forward {sl} {vo} {ed} {fd} {ah} {hd} {Γ} sc mask p wseq pr =
      let p' = block-param (wkgptp suc p) pr in
      -- embed
      Let seq := m-rmsnorm {sl} ((p .wpe) ⊞ wseq) In
      -- block into heads
      Let seq' := block-seq seq pr In
      -- layer pass
      Let lseq := mgpt-layer {ah = ah} sc ⟨ mask ⟩ p' seq' In
      -- decode into vocabulary
      Let logits := m-linear {u = vo} {p = sl} ⟨ p' .wvoc ⟩ lseq In logits

    cross-entropy : ∀ {Γ} (logits target : E Γ (ar s)) → (E Γ (ar []))
    cross-entropy {s} logits target =
      Let probs := softmax logits In
      Let lnprobs := ln probs In
      Let r := ⊟ (Sum λ i → sels lnprobs i ⊠ sels ⟨ target ⟩ i) In r

    m-cross-entropy : ∀ {Γ} (logits target : E Γ (ar (s ⊗ p))) → (E Γ (ar s))
    m-cross-entropy {s} {p} logits target =
      Imaps λ i → cross-entropy {p} (sel ⟨ logits ⟩ i) (sel ⟨ target ⟩ i)

    mgpt-loss : ∀ {Γ} (sc : ℕ) (mask : E Γ (ar (sl ⊗ sl)))
                (p : GPT-Params Γ vo ed sl fd) (wseq : E Γ (ar (sl ⊗ ed)))
                (target : E Γ (ar (sl ⊗ vo))) → ah * hd ≈ ed → E Γ (ar [])
    mgpt-loss {sl} {vo} {ed} {fd} {ah} {hd} sc mask p wseq target pr =
      Let logits := mgpt-forward sc mask p wseq pr In
      Let losses := m-cross-entropy {sl} logits ⟨ target ⟩ In
      Let loss := avg losses In loss

    ED = ι 16 ; AH = ι 4 ; HD = ι 4 ; SL = ι 16 ; FD = ι 64 ; SC = 2 ; VO = ι 27

    PR : AH * HD ≈ ED
    PR = cons

    test-sels : ∀ {Γ} (inp : E Γ (ar [])) → (E Γ (ar []))
    test-sels {s} inp = Imaps λ j → sels ⟨ inp ⟩ j

    test-let : ∀ {Γ} (inp : E Γ (ar [])) → (E Γ (ar []))
    test-let inp = Let a := (Let b := inp ⊞ one In b ⊞ one ⊞ one) In a ⊞ one ⊞ one ⊞ one 

    test2-let : E ε (ar [])
    test2-let = Let a := one {s = []} In zero

    test3-let : ∀ {Γ} (inp : E Γ (ar [])) → (E Γ (ar []))
    test3-let inp = Let a := (Let b := inp In b ⊠ b) In a ⊞ one
    
    test-sels-e : E _ _
    test-sels-e = Lcon (ar [] ∷ []) (ar []) ε λ x → test-sels x

    test-let-e : E _ _
    test-let-e = Lcon (ar [] ∷ []) (ar []) ε λ x → test-let x

    test3-let-e : E _ _
    test3-let-e = Lcon (ar [] ∷ []) (ar []) ε λ x → test3-let x

--     let′ (Lang.E.var Lang._∈_.here)
--    (let′ Lang.E.one
--    (let′ (Lang.E.var (Lang._∈_.there (Lang._∈_.there Lang._∈_.here)))
--    (let′ (Lang.E.var (Lang._∈_.there Lang._∈_.here))
--      (env (ε ▹ Lang.E.var Lang._∈_.here)))))

    -- test2-let : ∀ {Γ} (inp : E Γ (ar [])) → (E Γ (ar []))
    -- test2-let inp = {!   !}

    avg-e : E _ _
    avg-e = Lcon (ar SL ∷ []) (ar []) ε
      λ x → avg x

    m-softmax-e : E _ _
    m-softmax-e = Lcon (ar (SL ⊗ ED) ∷ []) (ar (SL ⊗ ED)) ε 
      λ x → m-softmax {SL} x
    
    cross-entropy-e : E _ _
    cross-entropy-e = Lcon (ar ED ∷ ar ED ∷ []) (ar []) ε
      λ x y → cross-entropy x y

    mgpt-forward-e : E _ _
    mgpt-forward-e = Lcon (ar (SL ⊗ SL) ∷ ar (SL ⊗ ED) ∷ ar (ED ⊗ ED) ∷
                  ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷
                  ar (FD ⊗ ED) ∷ ar (ED ⊗ FD) ∷ ar (VO ⊗ ED) ∷
                  ar (SL ⊗ ED) ∷ []) (ar (SL ⊗ VO)) ε
      λ mask wpe wqry wkey wval wout wup wdown wvoc wseq  →
        mgpt-forward {sl = SL} SC mask 
          (to-gptp wpe wqry wkey wval wout wup wdown wvoc) wseq PR
    
    mgpt-loss-e : E _ _
    mgpt-loss-e = Lcon (ar (SL ⊗ SL) ∷ ar (SL ⊗ ED) ∷ ar (ED ⊗ ED) ∷
                  ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷
                  ar (FD ⊗ ED) ∷ ar (ED ⊗ FD) ∷ ar (VO ⊗ ED) ∷
                  ar (SL ⊗ ED) ∷ ar (SL ⊗ VO) ∷ []) (ar []) ε
      λ mask wpe wqry wkey wval wout wup wdown wvoc wseq target →
        mgpt-loss {sl = SL} SC mask
          (to-gptp wpe wqry wkey wval wout wup wdown wvoc) wseq target PR

--     mgpt-loss-e : E _ _
--     mgpt-loss-e = Lcon (ar (SL ⊗ SL) ∷ ar (VS ⊗ ED) ∷ ar (SL ⊗ ED) ∷ ar (ED ⊗ ED) ∷
--                   ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷
--                   ar ((FD ⊗ ED) ⊗ ED) ∷ ar (ED ⊗ (FD ⊗ ED)) ∷ ar (VS ⊗ ED) ∷
--                   ar (SL ⊗ VS) ∷ ar (SL ⊗ VS) ∷ []) (ar []) ε
--       λ at-mask wte wpe wqry wkey wval wout wup wdown wvoc doc-id tar-id →
--         mgpt-loss {SL} {AH} {fd = FD} SC at-mask (record
--            { wte = wte
--            ; wpe = wpe
--            ; wqry = wqry
--            ; wkey = wkey
--            ; wval = wval
--            ; wout = wout
--            ; wup = wup
--            ; wdown = wdown
--            ; wvoc = wvoc
--            }) doc-id tar-id

--     mgpt-loss : ∀ {Γ} (sc : ℕ) (at-mask : E Γ (ar (sl ⊗ sl)))
--                 (p : GPT-Params Γ (ah ⊗ hd) sl fd vs)
--                 (doc-id tar-id : E Γ (ar (sl ⊗ vs))) → E Γ (ar [])
--     mgpt-loss {sl} {ah} {hd} {fd} {vs} {Γ} sc at-mask p doc-id tar-id =
--       Let logits := mgpt-forward {ah = ah} sc at-mask p doc-id In
--       Let losses := m-cross-entropy {sl} logits ⟨ tar-id ⟩ In
--       Let loss := avg losses In loss


    -- mgpt-layer : ∀ {Γ} (sc : ℕ) (mask : E Γ (ar (sl ⊗ sl)))
    --              (p : GPT-Params Γ {ah} {hd} {fd} vo ed sl pr)
    --              (seq : E Γ (ar (sl ⊗ ed))) → E Γ (ar (sl ⊗ ed))
    -- mgpt-layer {sl} {ah} {hd} {fd} {vo} {ed} {pr} sc mask p seq =
    --   Let nseq := m-rmsnorm {sl} seq In
    --   Let qs := m-linear {u = ed} {p = sl} ⟨ p .wqry ⟩ nseq In
    --   Let ks := m-linear {u = ed} {p = sl} ⟨ p .wkey ⟩ nseq In
    --   Let vs := m-linear {u = ed} {p = sl} ⟨ p .wval ⟩ nseq In {!   !}

--     mgpt-layer : let ed = ah ⊗ hd in
--                  ∀ {Γ} (sc : ℕ) (mask : E Γ (ar (sl ⊗ sl)))
--                  (p : GPT-Params Γ ed sl fd vs) (seq : E Γ (ar (sl ⊗ ed)))
--                  → E Γ (ar (sl ⊗ ed))
--     mgpt-layer {ah} {hd} {sl} {fd} {vs} {Γ} sc mask p seq = let ed = ah ⊗ hd in
--       Let nseq := m-rmsnorm {sl} seq In
--       Let qs := m-linear {u = ed} {p = sl} ⟨ p .wqry ⟩ nseq In
--       Let ks := m-linear {u = ed} {p = sl} ⟨ p .wkey ⟩ nseq In
--       Let vs := m-linear {u = ed} {p = sl} ⟨ p .wval ⟩ nseq In
--       Let attn := mh-attention {sl} {ah} {hd} sc ⟨ mask ⟩ qs ks vs In
--       Let oseq := m-linear {u = ed} {p = sl} ⟨ p .wout ⟩ attn In
--       Let cseq := oseq ⊞ ⟨ seq ⟩ In
--       Let fseq := mlp {fd = fd} {sl = sl} ⟨ p .wup ⟩ ⟨ p .wdown ⟩ cseq In fseq

--     embed-doc : ∀ {Γ} (p : GPT-Params Γ ed sl fd vs)
--                 (doc-id : E Γ (ar (sl ⊗ vs))) → E Γ (ar (sl ⊗ ed))
--     embed-doc {ed} {sl} {fd} {vs} {Γ} p doc-id =
--       matmul {sl} doc-id ⟨ p .wte ⟩ ⊞  (p .wpe)

--     mgpt-forward : ∀ {Γ} (sc : ℕ) (att-mask : E Γ (ar (sl ⊗ sl)))
--                    (p : GPT-Params Γ (ah ⊗ hd) sl fd vs)
--                    (doc-id : E Γ (ar (sl ⊗ vs))) → E Γ (ar (sl ⊗ vs))
--     mgpt-forward {sl} {ah} {hd} {fd} {vs} sc att-mask p doc-id =
--       -- Embedding
--       Let seq := embed-doc p doc-id In
--       -- layer pass
--       Let lseq := mgpt-layer {ah} sc ⟨ att-mask ⟩ (wkgptp p) seq In
--       -- project into vocabulary
--       Let logits := m-linear {u = vs} {p = sl} ⟨ p .wvoc ⟩ lseq In logits

--     cross-entropy : ∀ {Γ} (logits target : E Γ (ar s)) → (E Γ (ar []))
--     cross-entropy {s} logits target =
--       ⊟ Sum (λ i → (sels ⟨ target ⟩ i) ⊠ ln (softmax (sels ⟨ logits ⟩ i)))

--     m-cross-entropy : ∀ {Γ} (logits target : E Γ (ar (s ⊗ p))) → (E Γ (ar s))
--     m-cross-entropy {s} {p} logits target =
--       Imaps λ i → cross-entropy {p} (sel ⟨ logits ⟩ i) (sel ⟨ target ⟩ i)

--     mgpt-loss : ∀ {Γ} (sc : ℕ) (at-mask : E Γ (ar (sl ⊗ sl)))
--                 (p : GPT-Params Γ (ah ⊗ hd) sl fd vs)
--                 (doc-id tar-id : E Γ (ar (sl ⊗ vs))) → E Γ (ar [])
--     mgpt-loss {sl} {ah} {hd} {fd} {vs} {Γ} sc at-mask p doc-id tar-id =
--       Let logits := mgpt-forward {ah = ah} sc at-mask p doc-id In
--       Let losses := m-cross-entropy {sl} logits ⟨ tar-id ⟩ In
--       Let loss := avg losses In loss

--     AH = ι 4 ; HD = ι 4 ; SL = ι 16 ; FD = ι 4 ; SC = 2 ; VS = ι 27
--     ED = AH ⊗ HD ;  ED' = ι 16

--     -- test : ∀ {Γ} → E Γ (ar ED') → E Γ (ar ED)
--     -- test x = Imap {AH} λ i → selb cons ⟨ x ⟩ i

--     mgpt-loss-e : E _ _
--     mgpt-loss-e = Lcon (ar (SL ⊗ SL) ∷ ar (VS ⊗ ED) ∷ ar (SL ⊗ ED) ∷ ar (ED ⊗ ED) ∷
--                   ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷
--                   ar ((FD ⊗ ED) ⊗ ED) ∷ ar (ED ⊗ (FD ⊗ ED)) ∷ ar (VS ⊗ ED) ∷
--                   ar (SL ⊗ VS) ∷ ar (SL ⊗ VS) ∷ []) (ar []) ε
--       λ at-mask wte wpe wqry wkey wval wout wup wdown wvoc doc-id tar-id →
--         mgpt-loss {SL} {AH} {fd = FD} SC at-mask (record
--            { wte = wte
--            ; wpe = wpe
--            ; wqry = wqry
--            ; wkey = wkey
--            ; wval = wval
--            ; wout = wout
--            ; wup = wup
--            ; wdown = wdown
--            ; wvoc = wvoc
--            }) doc-id tar-id

--     -- mgpt-loss-e = Lcon (ar (VS ⊗ ED) ∷ ar (SL ⊗ ED) ∷ ar (ED ⊗ ED) ∷
--     --               ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷
--     --               ar ((FD ⊗ ED) ⊗ ED) ∷ ar (ED ⊗ (FD ⊗ ED)) ∷ ar (VS ⊗ ED) ∷
--     --               ar (SL ⊗ VS) ∷ ar (SL ⊗ VS) ∷ []) (ar []) ε

--     -- mgpt-loss {ah} {hd} {sl} {fd} {vs} {Γ} sc p doc-id target =
--     --   Let logits := mgpt-forward {ah} sc p doc-id In
--     --   Let losses := m-cross-entropy {sl} logits ⟨ target ⟩ In
--     --   Let loss := avg losses In loss

--     -- mgpt-forward : ∀ {Γ} (sc : ℕ) (p : GPT-Params Γ (ah ⊗ hd) sl fd vs)
--     --                (doc-id : E Γ (ar (sl ⊗ vs))) → E Γ (ar (sl ⊗ vs))
--     -- mgpt-forward {ah = ah} {hd = hd} {sl = sl} {vs = vs} sc p doc-ix =
--     --   -- layer pass
--     --   Let lseq := mgpt-layer {ah} sc p (embed-doc p doc-ix) In
--     --   -- project into vocabulary
--     --   Let logits := m-linear {u = vs} {p = sl} ⟨ p .wvoc ⟩ lseq In logits

--     -- mgpt-layer : let ed = ah ⊗ hd in
--     --              ∀ {Γ} (sc : ℕ) (p : GPT-Params Γ ed sl fd vs)
--     --              (seq : E Γ (ar (sl ⊗ ed))) → E Γ (ar (sl ⊗ ed))
--     -- mgpt-layer {ah} {hd} {sl} {fd} sc p seq = let ed = ah ⊗ hd in
--     --   -- Multi headed attention block
--     --   Let nseq := rmsnorm seq In --TODO: multiple norms
--     --   Let qs := m-linear {u = ed} {p = sl} ⟨ p .wqry ⟩ nseq In
--     --   Let ks := m-linear {u = ed} {p = sl} ⟨ p .wkey ⟩ nseq In
--     --   Let vs := m-linear {u = ed} {p = sl} ⟨ p .wval ⟩ nseq In
--     --   Let attn := m-attention {sl} {ah} sc qs ks vs In
--     --   Let oseq := m-linear {u = ed} {p = sl} ⟨ p .wout ⟩ attn In
--     --   Let cseq := oseq ⊞ ⟨ seq ⟩ In
--     --   -- MLP block
--     --   Let fseq := mlp {fd = fd} {sl = sl} ⟨ p .wup ⟩ ⟨ p .wdown ⟩ cseq In fseq

--     -- attention : ∀ {Γ} (sc : ℕ) (qs ks vs : E Γ (ar (sl ⊗ hd)))
--     --           → E Γ (ar (sl ⊗ hd))
--     -- attention {sl} {hd} sc qs ks vs = matmul {r = hd}
--     --   (softmax (scaledown sc ⟨ matmul {u = sl} qs (icom {sl} ks) ⟩)) vs

--     -- m-attention : let i = (sl ⊗ (ah ⊗ hd)) in
--     --               ∀ {Γ} (sc : ℕ) (qs ks vs : E Γ (ar i)) → E Γ (ar i)
--     -- m-attention {sl} {ah} {hd} sc qs ks vs =
--     --   Imap {sl} λ i → Imap {ah} λ j →
--     --     attention {[]} sc
--     --       (sel (sel ⟨ qs ⟩ i) j)
--     --       (sel (sel ⟨ ks ⟩ i) j)
--     --       (sel (sel ⟨ vs ⟩ i) j)

--     -- mlp : ∀ {Γ} (wup : E Γ (ar ((fd ⊗ ed) ⊗ ed))) (wdown : E Γ (ar (ed ⊗ (fd ⊗ ed))))
--     --       (seq : E Γ (ar (sl ⊗ ed))) → E Γ (ar (sl ⊗ ed))
--     -- mlp {fd} {ed} {sl} wup wdown seq =
--     --   Let nseq := rmsnorm seq In
--     --   Let useq := m-linear {u = fd ⊗ ed} ⟨ wup ⟩ nseq In
--     --   Let aseq := relu useq In
--     --   Let dseq := m-linear {u = ed} {p = sl} ⟨ wdown ⟩ aseq In
--     --   Let cseq := dseq ⊞ ⟨ seq ⟩ In cseq

--     -- -- TODO: add biases and masking
--     -- mgpt-layer : let ed = ah ⊗ hd in
--     --              ∀ {Γ} (sc : ℕ) (p : GPT-Params Γ ed sl fd vs)
--     --              (seq : E Γ (ar (sl ⊗ ed))) → E Γ (ar (sl ⊗ ed))
--     -- mgpt-layer {ah} {hd} {sl} {fd} sc p seq = let ed = ah ⊗ hd in
--     --   -- Multi headed attention block
--     --   Let nseq := rmsnorm seq In --TODO: multiple norms
--     --   Let qs := m-linear {u = ed} {p = sl} ⟨ p .wqry ⟩ nseq In
--     --   Let ks := m-linear {u = ed} {p = sl} ⟨ p .wkey ⟩ nseq In
--     --   Let vs := m-linear {u = ed} {p = sl} ⟨ p .wval ⟩ nseq In
--     --   Let attn := m-attention {sl} {ah} sc qs ks vs In
--     --   Let oseq := m-linear {u = ed} {p = sl} ⟨ p .wout ⟩ attn In
--     --   Let cseq := oseq ⊞ ⟨ seq ⟩ In
--     --   -- MLP block
--     --   Let fseq := mlp {fd = fd} {sl = sl} ⟨ p .wup ⟩ ⟨ p .wdown ⟩ cseq In fseq

--     -- mgpt-forward : ∀ {Γ} (sc : ℕ) (p : GPT-Params Γ (ah ⊗ hd) sl fd vs)
--     --                (doc-id : E Γ (ar (sl ⊗ vs))) → E Γ (ar (sl ⊗ vs))
--     -- mgpt-forward {ah = ah} {hd = hd} {sl = sl} {vs = vs} sc p doc-ix =
--     --   -- layer pass
--     --   Let lseq := mgpt-layer {ah} sc p (embed-doc p doc-ix) In
--     --   -- project into vocabulary
--     --   Let logits := m-linear {u = vs} {p = sl} ⟨ p .wvoc ⟩ lseq In logits

--     -- cross-entropy : ∀ {Γ} (logits target : E Γ (ar s)) → (E Γ (ar []))
--     -- cross-entropy {s} logits target =
--     --   ⊟ Sum (λ i → (sels ⟨ target ⟩ i) ⊠ ln (softmax (sels ⟨ logits ⟩ i)))

--     -- m-cross-entropy : ∀ {Γ} (logits target : E Γ (ar (s ⊗ p))) → (E Γ (ar s))
--     -- m-cross-entropy {s} {p} logits target =
--     --   Imaps λ i → cross-entropy {p} (sel ⟨ logits ⟩ i) (sel ⟨ target ⟩ i)

--     -- mgpt-loss : ∀ {Γ} → (sc : ℕ) (p : GPT-Params Γ (ah ⊗ hd) sl fd vs)
--     --             (doc-id target : E Γ (ar (sl ⊗ vs))) → E Γ (ar [])
--     -- mgpt-loss {ah} {hd} {sl} {fd} {vs} {Γ} sc p doc-id target =
--     --   Let logits := mgpt-forward {ah} sc p doc-id In
--     --   Let losses := m-cross-entropy {sl} logits ⟨ target ⟩ In
--     --   Let loss := avg losses In loss

--     -- AH = ι 4 ; HD = ι 4 ; SL = ι 16 ; FD = ι 4 ; SC = 2 ; VS = ι 27
--     -- ED = AH ⊗ HD

--     -- mgpt-loss-e : E _ _
--     -- mgpt-loss-e = Lcon (ar (VS ⊗ ED) ∷ ar (SL ⊗ ED) ∷ ar (ED ⊗ ED) ∷
--     --               ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷ ar (ED ⊗ ED) ∷
--     --               ar ((FD ⊗ ED) ⊗ ED) ∷ ar (ED ⊗ (FD ⊗ ED)) ∷ ar (VS ⊗ ED) ∷
--     --               ar (SL ⊗ VS) ∷ ar (SL ⊗ VS) ∷ []) (ar []) ε
--     --   λ wte wpe wqry wkey wval wout wup wdown wvoc doc-id target →
--     --     mgpt-loss {ah = AH} {hd = HD} {sl = SL} {fd = FD} SC (record
--     --        { wte = wte
--     --        ; wpe = wpe
--     --        ; wqry = wqry
--     --        ; wkey = wkey
--     --        ; wval = wval
--     --        ; wout = wout
--     --        ; wup = wup
--     --        ; wdown = wdown
--     --        ; wvoc = wvoc
--     --        }) doc-id target

--     -- -- cross-entropy : ∀ {Γ} (inp target : E Γ (ar s)) → (E Γ (ar []))
--     -- -- cross-entropy {s} inp target =
--     -- --   ⊟ (Sum (λ i → sels ⟨ target ⟩ i ⊠ ln (sels ⟨ softmax inp ⟩ i)))

--     -- embed-doc : (p : GPT-Params Γ ed sl fd vs) (doc-ix : E Γ (ix (sl ⊗ vs)))
--     --             → E Γ (ar (sl ⊗ ed))
--     -- embed-doc {Γ} {ed} {sl} {fd} {vs} p doc-ix =
--     --   Imap {sl} λ i → rmsnorm (sel ⟨ p .wpe ⟩ i ⊞ sel wtes ⟨ doc-ix ⟩) where
--     --   -- inefficient?
--     --   wtes = subst-assr {s = sl} (tile {p = sl} ⟨ p .wte ⟩)
--     -- mgpt-loss : ∀ {Γ} → (sc : ℕ) (p : GPT-Params Γ (ah ⊗ hd) sl fd vs)
--     --             (doc-ix target-ix : E Γ (ix (sl ⊗ vs))) → E Γ (ar [])
--     -- mgpt-loss {ah} {hd} {sl} {fd} {vs} {Γ} sc p doc-ix target-ix =
--     --   Let logits := mgpt-forward {ah} sc p doc-ix In {!   !}

--     -- gpt-layer : let ed = ah ⊗ hd in
--     --             (p : GPT-Params Γ ed sl fd vs)
--     --             (seq : E Γ (ar (sl ⊗ ed))) (sc : ℕ) → E Γ (ar (sl ⊗ ed))
--     -- gpt-layer {ah} {hd} {Γ} {sl} {fd} {vs} p seq sc = let ed = ah ⊗ hd in
--     --   -- Multi headed attention block
--     --   Let nseq := rmsnorm seq In
--     --   Let qs := m-linear {u = ed} {p = sl} ⟨ p .wqry ⟩ nseq In
--     --   Let ks := m-linear {u = ed} {p = sl} ⟨ p .wkey ⟩ nseq In
--     --   Let vs := m-linear {u = ed} {p = sl} ⟨ p .wval ⟩ nseq In
--     --   Let attn1 := m-attention {ah} {sl} {hd} sc
--     --     (icom3 {sl} {ah} qs) (icom3 {sl} {ah} ks) (icom3 {sl} {ah} vs) In
--     --   Let attn := icom3 {ah} {sl} attn1 In
--     --   Let oseq := m-linear {u = ed} {p = sl} ⟨ p .wout ⟩ attn In
--     --   Let cseq := oseq ⊞ ⟨ seq ⟩ In
--     --   -- MLP block
--     --   Let out := mlp {fd = fd} {sl = sl} ⟨ p .wup ⟩ ⟨ p .wdown ⟩ cseq In out

--     -- gpt-layer : let ed = ah ⊗ hd in
--     --             (p : GPT-Params Γ ed sl fd vs)
--     --             (seq : E Γ (ar (sl ⊗ ed))) (sc : ℕ) → E Γ (ar ed)
--     -- gpt-layer {ah} {hd} {Γ} {sl} {fd} {vs} p seq sc = let ed = ah ⊗ hd in
--     --   -- Multi headed attention block
--     --   Let nseq := rmsnorm seq In
--     --   Let qs := m-linear {u = ed} {s = ed} {p = sl} ⟨ p .wqry ⟩ nseq In
--     --   Let ks := m-linear {u = ed} {s = ed} {p = sl} ⟨ p .wkey ⟩ nseq In
--     --   Let vs := m-linear {u = ed} {s = ed} {p = sl} ⟨ p .wval ⟩ nseq In {!   !}

--     -- embed-seq : ∀ {Γ ed sl fd vs}
--     --             (p : GPT-Params Γ ed sl fd vs)
--     --             -- is this equivalent to ix sl ⊗ vs?
--     --             (doc-ix : E (Γ ▹ ix sl) (ix vs))
--     --             → E Γ (ar (sl ⊗ ed))
--     -- embed-seq {sl = sl} p doc-ix =
--     --   -- valid?
--     --   Imap {sl} λ i → rmsnorm (sel ⟨ p .wpe ⟩ i ⊞ sel ⟨ p .wte ⟩ doc-ix)

--     -- embed-tok : ∀ {Γ ed sl fd vs}
--     --             (tok-ix : E Γ (ix vs)) (pos-ix : E Γ (ix sl))
--     --             (p : GPT-Params Γ ed sl fd vs)
--     --             → E Γ (ar ed)
--     -- embed-tok tok-ix pos-ix p =
--     --   rmsnorm ((sel (p .wte) tok-ix) ⊞ sel (p .wpe) pos-ix)

--     -- embed-seq : ∀ {Γ ed sl fd vs}
--     --             (p : GPT-Params Γ ed sl fd vs) (doc-ix : E (Γ ▹ ix sl) (ix vs))
--     --             → E Γ (ar (sl ⊗ ed))
--     -- embed-seq {Γ} {ed} {sl} {fd} {vs} p doc-ix =
--     --   Imap {sl} λ i → embed-tok i {!   !} {!   !}


--     -- gpt-layer : ∀ {Γ ah hd sl fd vs} → let i = ah ⊗ hd in
--     --             (p : GPT-Params Γ ah hd sl fd vs) (tok_ix : E Γ (ix vs))
--     --             (pos_ix : E Γ (ix sl)) (sc : ℕ)
--     --             (keys vals : E Γ (ar (sl ⊗ ed)))
--     --             → E Γ (ar ed) × (E Γ (ar (sl ⊗ ed)) × E Γ (ar (sl ⊗ ed)))

--     -- update keys and values
--     -- We assume olds starts as zero
--     -- update : ∀ {Γ ed sl} →
--     --          (pos_ix : E Γ (ix sl)) (olds : E Γ (ar (sl ⊗ ed)))
--     --          (new : E Γ (ar ed)) → E Γ (ar (sl ⊗ ed))
--     -- update pos_ix olds new = olds ⊞ Imap λ i → zero-but ⟨ pos_ix ⟩ i ⟨ new ⟩

--     -- gpt-layer : ∀ {Γ ah hd sl fd vs} → let ed = ah ⊗ hd in
--     --             (p : GPT-Params Γ ah hd sl fd vs) (tok_ix : E Γ (ix vs))
--     --             (pos_ix : E Γ (ix sl)) (sc : ℕ)
--     --             (keys vals : E Γ (ar (sl ⊗ ed)))
--     --             → E Γ (ar ed) × (E Γ (ar (sl ⊗ ed)) × E Γ (ar (sl ⊗ ed)))
--     -- gpt-layer {ah = ah} {hd = hd} p tok_ix pos_ix sc keys vals =
--     --   logits , update pos_ix keys key , update pos_ix vals key where

--     --   emb = embed tok_ix pos_ix (p .wte) (p .wpe)

--     --   nemb = rmsnorm emb

--     --   qry = linear {u = ed p} (p .wqry) nemb
--     --   key = linear {u = ed p} (p .wkey) nemb
--     --   val = linear {u = ed p} (p .wval) nemb

--     --   logits = {!   !}



--     -- gpt-layer : ∀ {Γ ah hd sl fd vs} → let ed = ah ⊗ hd in
--     --             (p : GPT-Params Γ ah hd sl fd vs) (tok_ix : E Γ (ix vs))
--     --             (pos_ix : E Γ (ix sl)) (sc : ℕ)
--     --             → E Γ (ar ed)
--     -- gpt-layer = {!   !}
--     -- TODO : add biases, batch size and masking
--     -- gpt-layer : ∀ {Γ ah hd sl fd vs} →
--     --             let ed = ah ⊗ hd in
--     --             (inp : E Γ (ar ed)) (p : GPT-Params Γ ah hd sl fd vs)
--     --             (keys vals : E Γ (ar ed))
--     --              → E Γ (ar ed)
--     -- gpt-layer = {!   !}
--     -- gpt-layer {Γ} {ah} {hd} {sl} {fd} inp p keys vals =
--     --   Let ninp := rmsnorm inp In
--     --   Let q := linear ⟨ p .wqry ⟩ ninp In
--     --   Let k := linear ⟨ p .wkey ⟩ ninp In
--     --   Let v := linear ⟨ p .wval ⟩ ninp In {!   !}

--     -- attention : ∀ {Γ u s r t} → E Γ (ar (u ⊗ s)) → E Γ (ar (r ⊗ s))
--     --           → E Γ (ar (r ⊗ t)) → ℕ → E Γ (ar (u ⊗ t))
--     -- attention {Γ} {u} {s} {r} {t} q k v sc =
--     --   matmul {u}
--     --     (softmax (scaledown sc ⟨ matmul {u} {s} q (icom {r} k) ⟩)) ⟨ v ⟩

--     -- icom3 : ∀ {Γ} → E Γ (ar (p ⊗ (s ⊗ u))) → E Γ (ar (u ⊗ (p ⊗ s)))
--     -- icom3 {p} {s} {u} x = icoms {(p ⊗ s)} {u} (subst-assr {s = p} x)

--     -- -- Is this correct?
--     -- m-attention : let i = (ah ⊗ (sl ⊗ hd)) in
--     --               (sc : ℕ) (qs ks vs : E Γ (ar i)) → E Γ (ar i)
--     -- m-attention {ah} {sl} {hd} {Γ} sc qs ks vs =
--     --   Imap {ah} λ i →
--     --     attention {sl} sc (sel ⟨ qs ⟩ i) (sel ⟨ ks ⟩ i) (sel ⟨ vs ⟩ i)

--     -- m-attention : let i = (sl ⊗ (ah ⊗ hd)) in
--     --               (qs ks vs : E Γ (ar i)) (sc : ℕ) → E Γ (ar i)
--     -- m-attention {sattn3l} {ah} {hd} {Γ} qs ks vs sc =
--     --   icom3 {ah} {sl} (
--     --     Imap {ah} λ i → attention {sl} sc
--     --       (sel (icom3 {sl} {ah} ⟨ qs ⟩) i)
--     --       (sel (icom3 {sl} {ah} ⟨ qs ⟩) i)
--     --       (sel (icom3 {sl} {ah} ⟨ qs ⟩) i))

--     -- m-attention : let i = (sl ⊗ (ah ⊗ hd)) in
--     --               (qs ks vs : E Γ (ar i)) (sc : ℕ) → E Γ (ar i)
--     -- m-attention {sl} {ah} {hd} {Γ} qs ks vs sc = let
--     --   qs' = icom3 {sl} {hd} {ah} (icom {sl} {ah} qs)
--     --   ks' = icom3 {sl} {hd} {ah} (icom {sl} {ah} ks)
--     --   vs' = icom3 {sl} {hd} {ah} (icom {sl} {ah} vs)

--     --   in icom3 {ah} {hd} {sl} (icom {ah} {sl} (
--     --     Imap {ah} λ i → attention {sl} sc (sel ⟨ qs' ⟩ i) (sel ⟨ ks' ⟩ i) (sel ⟨ vs' ⟩ i)))

--       -- (Imap {ah} λ i →
--       --     attention {sl} {hd} sc (sel ⟨ qs' ⟩ i) (sel ⟨ ks' ⟩ i) (sel ⟨ vs' ⟩ i))

--         -- ( icom {ah} {sl} (
--         -- Imap {ah} λ i → attention {sl} sc (sel {! swa  !} i) {!   !} {!   !}))

--       -- subst-assl {sl} {hd} (icoms {ah} (
--       --   Imap {ah} λ i → attention {sl}
--       --     (sel (icom3 {sl} {hd} ⟨ qs ⟩) i)
--       --     (sel (icom3 {sl} {hd} ⟨ ks ⟩) i)
--       --     (sel (icom3 {sl} {hd} ⟨ vs ⟩) i) sc))

--       -- m-attention : ∀ {h u s r t Γ} → E Γ (ar (h ⊗ (u ⊗ s)))
--     --           → E Γ (ar (h ⊗ (r ⊗ s))) → E Γ (ar (h ⊗ (r ⊗ t)))
--     --           → ℕ
--     --           → E Γ (ar (h ⊗ (u ⊗ t)))
--     -- m-attention {h} {u} q k v sc =
--     --   Imap {h} (λ i →
--     --     attention {u = u} (sel ⟨ q ⟩ i) (sel ⟨ k ⟩ i) (sel ⟨ v ⟩ i) sc)

--     -- module Microgpt where

--   --   linear : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar s) → E Γ (ar u)
--   --   linear {u} {s} w x =
--   --     Imaps {u} λ i → Sum {s} λ j → sels (sel ⟨ w ⟩ i) j ⊠ sels ⟨ x ⟩ j

--   --   matmul : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (s ⊗ r)) → E Γ (ar (u ⊗ r))
--   --   matmul {u} {s} {r} w x = Imap {u} λ i →
--   --     Imaps (λ j → sels (linear ⟨ w ⟩ (Imaps λ k → sels (sel ⟨ x ⟩ k) j) ) i)

--   --   -- Is this correct?
--   --   softmax : ∀ {Γ} → E Γ (ar s) → E Γ (ar s)
--   --   softmax {s = s} x =
--   --       Imaps (λ i → (𝕖^ (sels ⟨ x ⟩ i)) // Sum (λ j → 𝕖^ sels ⟨ x ⟩ j))

--   --   -- add a small number to avoid dividing by zero?
--   --   rmsnorm : ∀ {Γ} → E Γ (ar s) → E Γ (ar s)
--   --   rmsnorm {s = s} x =
--   --     x ⊠ 𝟙/ (sqrt (scaledown (len s) (sum {s = s} (⟨ x ⟩ ⊠ ⟨ x ⟩))))

--   --   swap : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (s ⊗ u))
--   --   swap {u} {s} x = Imap {s} λ i → Imaps λ j → sels (sel ⟨ x ⟩ j) i

--   --   max : ∀ {Γ} → E Γ (ar s) → E Γ (ar s) → E Γ (ar s)
--   --   max x y = x ⊞ relu (y ⊟ x)

--   --   {- I cheat by passing the scale sc as a parameter. It should be such that
--   --     sqrt (size v) =  sc
--   --     For microgpt sc = 16.
--   --     Unlike microgpt, WE DO NOT MASK
--   --     TODO : figure out how to mask
--   --   -}
--   --   attention : ∀ {u s r t Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (r ⊗ s))
--   --             → E Γ (ar (r ⊗ t)) → ℕ → E Γ (ar (u ⊗ t))
--   --   attention {u} {s} {r} {t} q k v sc =
--   --     matmul {u}
--   --       (softmax (scaledown sc ⟨ matmul {u} {s} q (swap {r} k) ⟩)) ⟨ v ⟩

--   --   -- attention : ∀ {u s r t Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (r ⊗ s))
--   --   --           → E Γ (ar (r ⊗ t)) → ℕ → E Γ (ar (u ⊗ t))
--   --   -- attention {u} {s} {r} {t} q k v sc =
--   --   --   Let l₁ := matmul {u} {s} q (swap {r} k) In
--   --   --   Let l := scaledown sc l₁ In
--   --   --   Let w₁ := softmax l In
--   --   --   Let w := matmul {u} w₁ ⟨ v ⟩ In w

--   --   mattention : ∀ {h u s r t Γ} → E Γ (ar (h ⊗ (u ⊗ s)))
--   --             → E Γ (ar (h ⊗ (r ⊗ s))) → E Γ (ar (h ⊗ (r ⊗ t)))
--   --             → ℕ
--   --             → E Γ (ar (h ⊗ (u ⊗ t)))
--   --   mattention {h} {u} q k v sc =
--   --     Imap {h} (λ i →
--   --       attention {u} (sel ⟨ q ⟩ i) (sel ⟨ k ⟩ i) (sel ⟨ v ⟩ i) sc)

--   --   -- TODO: generalize to add biases and batch size
--   --   -- TODO: add masking
--   --   gpt-layer : ∀ {Γ ah hd sl fd} →
--   --                let i = ah ⊗ (hd ⊗ sl) in
--   --                (inp : E Γ (ar i)) (wq wk wv wo : E Γ (ar (i ⊗ i)))
--   --                (wf₁ : E Γ (ar ((fd ⊗ i) ⊗ i)))
--   --                (wf₂ : E Γ (ar (i ⊗ (fd ⊗ i))))
--   --                (sc : ℕ)
--   --                → E Γ (ar i)
--   --   gpt-layer {Γ} {ah} {hd} {sl} {fd} inp wq wk wv wo wf₁ wf₂ sc =
--   --               Let ninp := rmsnorm inp In
--   --               Let q := linear ⟨ wq ⟩ ninp In
--   --               Let k := linear ⟨ wk ⟩ ninp In
--   --               Let v := linear ⟨ wv ⟩ ninp In
--   --               Let c₁ :=
--   --                 mattention {ah} {hd} {sl} {hd}
--   --                 q k v sc In
--   --               Let s₁₁ := linear ⟨ wo ⟩ c₁ In
--   --               Let s₁ := s₁₁ ⊞ ⟨ inp ⟩ In
--   --               Let s₂₁ := rmsnorm s₁ In
--   --               Let s₂₂ := linear ⟨ wf₁ ⟩ s₂₁ In
--   --               Let s₂ := relu s₂₂ In
--   --               Let c₃ := linear ⟨ wf₂ ⟩ s₂ In
--   --               Let r := c₃ ⊞ s₁ In r

--   --   avg : ∀ {Γ} → E Γ (ar s) → E Γ (ar [])
--   --   avg {s} x = scaledown (len s) (Sum λ i → sels ⟨ x ⟩ i)

--   --   cross-entropy : ∀ {Γ} (inp target : E Γ (ar s)) → (E Γ (ar []))
--   --   cross-entropy {s} inp target =
--   --     ⊟ (Sum (λ i → sels ⟨ target ⟩ i ⊠ ln (sels ⟨ softmax inp ⟩ i)))

--   --   ED = 16 ; AH = 4 ; HD = ED ℕ./ AH ; SL = 16 ; FD = 4 ; SC = 2 ; VS = 27

--   --   W = (ι HD) ⊗ (ι SL)
--   --   I = (ι AH) ⊗ W

--   --   -- we calculate sequences in parallel (?)
--   --   microgpt : ∀ {Γ} →
--   --                (inp : E Γ (ar I)) (wq wk wv wo : E Γ (ar (I ⊗ I)))
--   --                (wf₁ : E Γ (ar ((ι FD ⊗ I) ⊗ I)))
--   --                (wf₂ : E Γ (ar (I ⊗ (ι FD ⊗ I))))
--   --                (sc : ℕ) (wvo : E Γ (ar (ι VS ⊗ I)))
--   --                → E Γ (ar (ι VS))
--   --   microgpt inp wq wk wv wo wf₁ wf₂ sc wvo =
--   --     Let s := gpt-layer {ah = ι AH } {hd = ι HD} {fd = ι FD}
--   --       inp wq wk wv wo wf₁ wf₂ SC In
--   --     -- normalize?
--   --     Let r := matmul {ι VS} ⟨ wvo ⟩ s In r

--   --   microgpt-token-loss : ∀ {Γ} →
--   --                (inp : E Γ (ar I)) (wq wk wv wo : E Γ (ar (I ⊗ I)))
--   --                (wf₁ : E Γ (ar ((ι FD ⊗ I) ⊗ I)))
--   --                (wf₂ : E Γ (ar (I ⊗ (ι FD ⊗ I))))
--   --                (sc : ℕ) (wvo : E Γ (ar (ι VS ⊗ I)))
--   --                (target : E Γ (ar (ι VS))) → E Γ (ar [])
--   --   microgpt-token-loss inp wq wk wv wo wf₁ wf₂ sc wvo target =
--   --     Let probs := microgpt inp wq wk wv wo wf₁ wf₂ sc wvo In
--   --     Let loss := cross-entropy probs ⟨ target ⟩ In loss

--     -- microgpt-loss : ∀ {Γ} →
--     --              (inp : E Γ (ar I)) (wq wk wv wo : E Γ (ar (I ⊗ I)))
--     --              (wf₁ : E Γ (ar ((ι FD ⊗ I) ⊗ I)))
--     --              (wf₂ : E Γ (ar (I ⊗ (ι FD ⊗ I))))
--     --              (sc : ℕ) (wvo : E Γ (ar (ι VS ⊗ I)))
--     --              (target_ix : E Γ (ix (ι VS))) → E Γ (ar [])
--     -- microgpt-loss inp wq wk wv wo wf₁ wf₂ sc wvo target_ix =
--     --   Let probs := microgpt inp wq wk wv wo wf₁ wf₂ sc wvo In
--     --   Let loss := cross-entropy probs ⟨ target_ix ⟩ In loss

--     -- microgpt : E _ _
--     -- microgpt =  Lcon (  ar I
--     --                   ∷ ar (I ⊗ I) ∷ ar (I ⊗ I) ∷ ar (I ⊗ I) ∷ ar (I ⊗ I)
--     --                   ∷ ar ((ι FD ⊗ I) ⊗ I) ∷ ar (I ⊗ (ι FD ⊗ I))
--     --                   ∷ ar ((ι VS) ⊗ (ι AH ⊗ ι HD)) ∷ [])
--     --                 (ar O) ε
--     --             λ inp wq wk wv wo wf₁ wf₂ wvo →
--     --             Let s := gpt-layer {ah = ι AH } {hd = ι HD} {fd = ι FD} inp wq wk wv wo wf₁ wf₂ SC In
--     --             -- normalize?
--     --             Let c := matmul {ι VS} {ι AH ⊗ ι HD} wvo s In
--     --             Let r := softmax c In r

--     -- microgpt : E _ _
--     -- microgpt =  Lcon (  ar I
--     --                   ∷ ar (I ⊗ I) ∷ ar (I ⊗ I) ∷ ar (I ⊗ I) ∷ ar (I ⊗ I)
--     --                   ∷ ar ((FD ∷ [] ⊗ I) ⊗ I) ∷ ar (I ⊗ (FD ∷ [] ⊗ I))
--     --                   ∷ [])
--     --                 (ar I) ε
--     --             λ inp wq wk wv wo wf₁ wf₂ →
--     --             Let ninp := rmsnorm inp In
--     --             Let q := linear wq ninp In
--     --             Let k := linear wk ninp In
--     --             Let v := linear wv ninp In
--     --             Let c₁ :=
--     --               mattention {AH ∷ []} {HD ∷ []} {SL ∷ []} {HD ∷ []}
--     --               q k v SC In
--     --             Let s₁₁ := linear wo c₁ In
--     --             Let s₁ := s₁₁ ⊞ inp In
--     --             Let s₂₁ := rmsnorm s₁ In
--     --             Let s₂₂ := linear wf₁ s₂₁ In
--     --             Let s₂ := relu s₂₂ In
--     --             Let c₃ := linear wf₂ s₂ In
--     -- --             Let r := c₃ ⊞ s₁ In r

--     -- attention-e : E _ _
--     -- attention-e =
--     --   Lcon (ar W ∷ ar W ∷ ar W ∷ []) (ar W) ε
--     --   λ q k v →
--     --     attention {HD ∷ []} {SL ∷ []} {HD ∷ []} q k v SC

--     -- triangular matrix experiment
--     -- id : ∀ {Γ} a → E Γ (ar (a ⊗ a))
--     -- id a = Imap {a} λ i → Imaps λ j → zero-but i j one

--     -- ex-id : ∀ {Γ} → E Γ (ar (ι 3 ⊗ ι 5))
--     -- ex-id = Imap {ι 3} λ i → E.backslide {p = ι 2} i (one {s = ι 3}) cons cons

--     -- c0=1 : ∀ {Γ} → E Γ (ar (ι 3 ⊗ ι 3))
--     -- c0=1 = Imap {ι 3} λ i → slide {p = ι 2} i cons (sel (ex-id) i) cons

--     -- r0=1 : ∀ {Γ} → E Γ (ar (ι 3 ⊗ ι 3))
--     -- r0=1 = swap {u = ι 3} c0=1

--     -- c1-2=1 : ∀ {Γ} → E Γ (ar (ι 3 ⊗ ι 3))
--     -- c1-2=1 = one ⊟ c0=1

--     -- r1-2=1 : ∀ {Γ} → E Γ (ar (ι 3 ⊗ ι 3))
--     -- r1-2=1 = swap {u = ι 3} c1-2=1

--     -- v-i : ∀ {Γ} → E Γ (ar (ι 3))
--     -- v-i = scaledown 3 (linear r0=1 one)

--     -- v1-2=1 : ∀ {Γ} → E Γ (ar (ι 3))
--     -- v1-2=1 = one ⊟ v-i

--     -- tile-r : ∀ {Γ s} → E Γ (ar s) → E Γ (ar (s ⊗ s))
--     -- tile-r {s = s} x = Imap {s} λ i → ⟨ x ⟩

--     -- triangular3 : ∀ {Γ} → E Γ (ar (ι 3 ⊗ ι 3))
--     -- triangular3 = c0=1 ⊞ {!   !}

-- module LangTest where
--   open import Ar
--   open import Data.List as L using (List; []; _∷_)
--   open import Function
--   open Syntax

--   nested-inc : E (Γ ▹ ar (s ⊗ p) ▹ ar p) (ar (s ⊗ p))
--   nested-inc {s = s} = imap {s = s} ((var v₁) ⊞ sel (var v₂) (var v₀))

--   -- Test convenience
--   _ : Prefix (Γ ▹ ar []) (Γ ▹ ar [] ▹ (ar (5 ∷ [])))
--   _ = it

--   _ : E Γ (ar (5 ∷ 5 ∷ []))
--   _ = Imaps λ iv → sels zero iv

--   _ : E Γ (ar (5 ∷ 5 ∷ []))
--   _ = Let x := zero In x ⊞ x

--   _ : E _ _
--   _ = Lcon (ar (5 ∷ []) ∷ ar [] ∷ []) (ar (5 ∷ [])) ε
--       λ a x → Let b := a ⊞ a In
--               Let c := (Imaps λ i → sel a i ⊠ x) In
--               c ⊞ c

-- -- automatically from other frameworks into ours?