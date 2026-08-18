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

    bin        : Bop → E Γ (ar s) → E Γ (ar s) → E Γ (ar s)
    scaledown  : ℕ → E Γ (ar s) → E Γ (ar s)
    let′       : E Γ (ar s) → E (Γ ▹ ar s) (ar p) → E Γ (ar p)
    -- Jairo made
    un         : Uop → E Γ (ar s) → E Γ (ar s)
    maximum    : E (Γ ▹ ix s) (ar p) → E Γ (ar p)

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

  𝕀0- : (E Γ (ar s)) → E Γ (ar s)
  𝕀0- a = 𝟙 ⊟ 𝕀+ a

  𝕀0+ : (E Γ (ar s)) → E Γ (ar s)
  𝕀0+ a = 𝕀0- (⊟ a)

  𝕀≤ : (E Γ (ar s)) → (E Γ (ar s)) → E Γ (ar s)
  𝕀≤ a b = 𝕀0+ (a ⊟ b)

  𝟚 : E Γ (ar s)
  𝟚 = 𝟙 ⊞ 𝟙

  var-inj : ∀ {x y : is ∈ Γ} → (var x ≡ var y) → (x ≡ y)
  var-inj refl = refl

module WkSub where
  open import Data.Nat using (ℕ; zero; suc; _+_)
  open import Relation.Binary.PropositionalEquality
  open import Function
  open import Ar hiding (sum; slide; backslide; map ; imapb; selb)

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
  wk s (bin x e e₁) = bin x (wk s e) (wk s e₁)
  wk s (scaledown x e) = scaledown x (wk s e)
  wk s (let′ e e₁) = let′ (wk s e) (wk (keep s) e₁)
  -- Jairo made
  wk s (un x e) = un x (wk s e)
  wk s (maximum e) = maximum (wk (keep s) e)

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

  -- ⊆-swap : (Γ ▹ is ▹ ip) ⊆ (Γ ▹ ip ▹ is)
  -- ⊆-swap {ε} {is} {ip} = keep {!   !} ∙ʷ {!   !}
  -- ⊆-swap {Γ ▹ x} {is} {ip} = {!   !}
    -- keep (keep (⊆-inj $ ⊆-inj s))

  -- ⊆-inj-ix : (Γ ▹ is) ⊆ (Γ ▹ ip) → is ≡ ip
  -- ⊆-inj-ix {ε} (keep s) = refl
  -- ⊆-inj-ix {Γ ▹ x} {is} {ip} s = ⊆-inj-ix {Γ} {!   !}

  data Sub (Γ : Ctx) : Ctx → Set where
    ε   : Sub Γ ε
    _▹_ : Sub Γ Δ → E Γ is → Sub Γ (Δ ▹ is)

  wks : Sub Γ Δ → Γ ⊆ Ψ → Sub Ψ Δ
  wks ε p = ε
  wks (s ▹ x) p = (wks s p) ▹ wk p x

  -- wkv-there-keep : (x : is ∈ Γ) (v : is ∈ Γ)
  --   → wkv (keep (wk-/ x)) v ≡ there (wkv (wk-/ x) v)
  -- wkv-there-keep = ?
  -- b : there y ≡ wkv (keep (wk-/ x)) a


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
  sub (bin x e e₁) s = bin x (sub e s) (sub e₁ s)
  sub (scaledown x e) s = scaledown x (sub e s)
  sub (let′ e e₁) s = let′ (sub e s) (sub e₁ (skeep s))
  -- Jairo made
  sub (un x e) s = un x (sub e s)
  sub (maximum e) s = maximum (sub e (skeep s))

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
  sub-at-id (bin x e e₁) = cong₂ (bin x) (sub-at-id e) (sub-at-id e₁)
  sub-at-id (scaledown x e) = cong (scaledown x) (sub-at-id e)
  sub-at-id (let′ e e₁) = cong₂ let′ (sub-at-id e) (sub-at-id e₁)
  -- Jairo made
  sub-at-id (un x e) = cong (un x) (sub-at-id e)
  sub-at-id (maximum e) = cong maximum (sub-at-id e)

  sub-ε : (e : E ε is) → sub e ε ≡ e
  sub-ε e = sub-at-id e

  sub-swap : Sub (Γ ▹ is ▹ ip) (Γ ▹ ip ▹ is)
  sub-swap = (sdrop (sdrop sub-id) ▹ var v₀) ▹ var (there v₀)

  -- We are not really using this, but this is a useful function to have.
  open import Data.Maybe
  open import Data.Maybe.Properties
  open import Data.Product hiding (map)

  -- strenv : (x : is ∈ Γ) (y : ip ∈ Γ) → Maybe (ip ∈ (Γ / x))
  -- strenv v₀ v₀ = nothing
  -- strenv v₀ (there y) = just y
  -- strenv (there x) v₀ = just v₀
  -- strenv (there x) (there y) = map there (strenv x y)

  -- stren : E Γ is → (v : ip ∈ Γ) → Maybe (E (Γ / v) is)
  -- stren (var x) v = map var (strenv v x)
  -- stren zero v = just zero
  -- stren one v = just one
  -- stren (imaps e) v = map imaps (stren e (there v))
  -- stren (sels e e₁) v = do
  --   l ← stren e v
  --   r ← stren e₁ v
  --   just (sels l r)
  -- stren (imap e) v = map imap (stren e (there v))
  -- stren (sel e e₁) v = do
  --   l ← stren e v
  --   r ← stren e₁ v
  --   just (sel l r)
  -- stren (imapb x e) v = map (imapb x) (stren e (there v))
  -- stren (selb x e e₁) v = do
  --   l ← stren e v
  --   r ← stren e₁ v
  --   just (selb x l r)
  -- stren (sum e) v = map sum (stren e (there v))
  -- stren (zero-but e e₁ e₂) v = do
  --   a ← stren e v
  --   b ← stren e₁ v
  --   c ← stren e₂ v
  --   just (zero-but a b c)
  -- stren (slide e x e₁ x₁) v = do
  --   a ← stren e v
  --   b ← stren e₁ v
  --   just (slide a x b x₁)
  -- stren (backslide e e₁ x x₁) v = do
  --   a ← stren e v
  --   b ← stren e₁ v
  --   just (backslide a b x x₁)
  -- stren (bin x e e₁) v = do
  --   a ← stren e v
  --   b ← stren e₁ v
  --   just (bin x a b)
  -- stren (scaledown x e) v = map (scaledown x) (stren e v)
  -- stren (⊟ e) v = map ⊟_ (stren e v)
  -- stren (let′ e e₁) v = do
  --   a ← stren e v
  --   b ← stren e₁ (there v)
  --   just (let′ a b)
  -- -- Jairo made
  -- stren (un x e) v = map (un x) (stren e v)
  -- stren (maximum e) v = map maximum (stren e (there v))

  -- from-stren : E Γ is → (v : ip ∈ Γ) → E (Γ / v) is → E (Γ / v) is
  -- from-stren e v e' = fromMaybe e' (stren e v)

  -- strenv-inj₂ : ∀ {iq} (x : is ∈ Γ) {y : ip ∈ Γ} {y' : ip ∈ (Γ / x)}
  --     → strenv (there {ip = iq} x) (there y) ≡ just (there y')
  --     → strenv x y ≡ just y'
  -- strenv-inj₂ (there x) {v₀} {v₀} eq = refl
  -- strenv-inj₂ v₀ {there y} {y'} eq = cong just (v-inj (just-injective eq))
  -- strenv-inj₂ (there x) {there y} {y'} eq with strenv x y | eq
  -- ... | just a | b = cong just (v-inj (just-injective b))

  -- var-stren-strenv : ∀ (x : is ∈ Γ) {y : ip ∈ Γ} {y' : ip ∈ (Γ / x)}
  --   → stren (var y) x ≡ just (var y') → strenv x y ≡ just y'
  -- var-stren-strenv v₀ {there y} {y'} refl = refl
  -- var-stren-strenv (there x) {v₀} {v₀} eq = refl
  -- var-stren-strenv (there x) {there y} {y'} eq with strenv x y | eq
  -- ... | just a | refl = refl

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
  stren-∃ zero v = just (zero , refl)
  stren-∃ one v = just (one , refl)
  stren-∃ (imaps e) v =
    map (λ (a , b) → _ , (cong imaps b)) (stren-∃ e (there v))
  stren-∃ (sels e e₁) v = do
    (a , b) ← stren-∃ e v
    (c , d) ← stren-∃ e₁ v
    just (_ , (cong₂ sels b d))
  stren-∃ (imap e) v =
    map (λ (a , b) → _ , (cong imap b)) (stren-∃ e (there v))
  stren-∃ (sel e e₁) v = do
    (a , b) ← stren-∃ e v
    (c , d) ← stren-∃ e₁ v
    just (_ , (cong₂ sel b d))
  stren-∃ (imapb x e) v =
    map (λ (a , b) → _ , (cong (E.imapb x) b)) (stren-∃ e (there v))
  stren-∃ (selb x e e₁) v = do
    (a , b) ← stren-∃ e v
    (c , d) ← stren-∃ e₁ v
    just (_ , (cong₂ (E.selb x) b d))
  stren-∃ (sum e) v =
    map (λ (a , b) → _ , (cong E.sum b)) (stren-∃ e (there v))
  stren-∃ (zero-but e e₁ e₂) v = do
    (a , b) ← stren-∃ e v
    (c , d) ← stren-∃ e₁ v
    (e , f) ← stren-∃ e₂ v
    just (_ , cong₃ zero-but b d f)
  stren-∃ (slide e x e₁ x₁) v = do
    (a , b) ← stren-∃ e v
    (c , d) ← stren-∃ e₁ v
    just (_ , (cong₂ (λ f g → E.slide f x g x₁) b d))
  stren-∃ (backslide e e₁ x x₁) v = do
    (a , b) ← stren-∃ e v
    (c , d) ← stren-∃ e₁ v
    just (_ , (cong₂ (λ f g → E.backslide f g x x₁) b d))
  stren-∃ (bin x e e₁) v = do
    (a , b) ← stren-∃ e v
    (c , d) ← stren-∃ e₁ v
    just (_ , (cong₂ (bin x) b d))
  stren-∃ (scaledown x e) v =
    map (λ (a , b) → _ , (cong (scaledown x) b)) (stren-∃ e v)
  stren-∃ (un x e) v =
    map (λ (a , b) → _ , (cong (un x) b)) (stren-∃ e v)
  stren-∃ (maximum e) v =
    map (λ (a , b) → _ , (cong E.maximum b)) (stren-∃ e (there v))
  stren-∃ (let′ e e₁) v = do
    (a , b) ← stren-∃ e v
    (c , d) ← stren-∃ e₁ (there v)
    just (_ , (cong₂ let′ b d))

  stren : (e : E Γ is) (v : ip ∈ Γ)
    → Maybe (E (Γ / v) is)
  stren e v = do
    (a , _) ← (stren-∃ e v)
    just a

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
  norm-lets (bin x e e₁) = bin x (norm-lets e) (norm-lets e₁)
  norm-lets (scaledown x e) = scaledown x (norm-lets e)
  norm-lets (let′ e e₁) = maybe id (let′ (norm-lets e) (norm-lets e₁)) (stren (norm-lets e₁) v₀)
  -- Jairo made
  norm-lets (un x e) = un x (norm-lets e)
  norm-lets (maximum e) = maximum (norm-lets e)

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
  count-uses (bin x e e₁) v = count-uses e v + count-uses e₁ v
  count-uses (scaledown x e) v = count-uses e v
  count-uses (let′ e e₁) v = count-uses e v + count-uses e₁ (there v)
  -- Jairo made
  count-uses (un x e) v = count-uses e v
  count-uses (maximum e) v = count-uses e (there v)

  --   count-sels' : E Γ is → ip ∈ Γ → ℕ → ℕ
  --   count-sels' (sels e (var i)) v count with eq? i v
  --   ... | veq = (count-sels' e v count) + 1
  --   ... | neq .i y = count
  --   count-sels' (sel e (var i)) v count with eq? i v
  --   ... | veq = (count-sels' e v count) + 1
  --   ... | neq .i y = count
  --   count-sels' (selb x e (var i)) v count with eq? i v
  --   ... | veq = (count-sels' e v count) + 1
  --   ... | neq .i y = count
  --   count-sels' e v count = count

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
  --   inline' (imapb x e) = imapb x (inline' e)
  --   inline' (selb x e e₁) = selb x (inline' e) (inline' e₁)
  --   inline' (sum e) = sum (inline' e)
  --   inline' (zero-but e e₁ e₂) = (zero-but (inline' e) (inline' e₁) (inline' e₂))
  --   inline' (slide e x e₁ x₁) = slide (inline' e) x (inline' e₁) x₁
  --   inline' (backslide e e₁ x x₁) = backslide (inline' e) (inline' e₁) x x₁
  --   inline' (bin x e e₁) = bin x (inline' e) (inline' e₁)
  --   inline' (scaledown x e) = scaledown x (inline' e)
  --   inline' (un x e) = un x (inline' e)
  --   inline' (maximum e) = maximum (inline' e)
  --   inline' (let′ e e₁) with a ← (inline' e₁) | count-uses a v₀ | e
  --   ... | 0 | _ = sub a (sub-id ▹ (inline' e))
  --   ... | _ | var b = sub a (sub-id ▹ (var b))
  --   ... | _ | zero = sub a (sub-id ▹ zero)
  --   ... | _ | one = sub a (sub-id ▹ one)
  --   -- ... | 1 | (imap b) = {!   !}
  --   -- ... | 1 | _ = sub a (sub-id ▹ (inline' e))
  --   ... | _ | _ = let′ (inline' e) a

  -- e-map : {Γ : Ctx} → (∀ {Δ ip} → Γ ⊆ Δ → E Δ ip → E Δ ip) → E Γ is → E Γ is
  -- e-map f (imaps e) = f ⊆-eq (imaps (e-map (λ s x → f (s ∙ʷ skip ⊆-eq) x) e))
  -- e-map f (imap e) = f ⊆-eq (imap (e-map (λ s x → f (s ∙ʷ skip ⊆-eq) x) e))
  -- e-map f (imapb x e) = f ⊆-eq (imapb x (e-map (λ s x → f (s ∙ʷ skip ⊆-eq) x) e))
  -- e-map f (sum e) = f ⊆-eq (sum (e-map (λ s x → f (s ∙ʷ skip ⊆-eq) x) e))
  -- e-map f (maximum e) = f ⊆-eq (maximum (e-map (λ s x → f (s ∙ʷ skip ⊆-eq) x) e))
  -- e-map f (sels e e₁) = f ⊆-eq (sels (e-map f e) (e-map f e₁))
  -- e-map f (sel e e₁) = f ⊆-eq (sel (e-map f e) (e-map f e₁))
  -- e-map f (selb x e e₁) = f ⊆-eq (selb x (e-map f e) (e-map f e₁))
  -- e-map f (zero-but e e₁ e₂) = f ⊆-eq (zero-but (e-map f e) (e-map f e₁) (e-map f e₂))
  -- e-map f (slide e x e₁ x₁) = f ⊆-eq (slide (e-map f e) x (e-map f e₁) x₁)
  -- e-map f (backslide e e₁ x x₁) = f ⊆-eq (backslide (e-map f e) (e-map f e₁) x x₁)
  -- e-map f (bin x e e₁) = f ⊆-eq (bin x (e-map f e) (e-map f e₁))
  -- e-map f (scaledown x e) = f ⊆-eq (scaledown x (e-map f e))
  -- e-map f (let′ e e₁) = f ⊆-eq (let′ (e-map f e) (e-map (λ s x → f (s ∙ʷ skip ⊆-eq) x) e₁))
  -- e-map f (un x e) = f ⊆-eq (un x (e-map f e))
  -- e-map f e = f ⊆-eq e

  -- e-map-∃ : {Γ : Ctx} {A : ∀ {Δ ip} → E Δ ip → Set}
  --   → (∀ {Δ ip} → (e : E Δ ip) → A e)
  --   → (e : E Γ is) → ∃ (λ (x : E Γ is) → A x)
  -- -- e-map-∃ f (imaps e) = _ , f (imaps (proj₁ (e-map-∃ f e)))
  -- e-map-∃ f (imaps e) = _ , f (imaps (proj₁ (e-map-∃ f e)))
  -- e-map-∃ f (imap e) = _ , f (imap (proj₁ (e-map-∃ f e)))
  -- e-map-∃ f (imapb x e) = _ , f (imapb x (proj₁ (e-map-∃ f e)))
  -- e-map-∃ f (sum e) = _ , f (sum (proj₁ (e-map-∃ f e)))
  -- e-map-∃ f (maximum e) = _ , f (maximum (proj₁ (e-map-∃ f e)))
  -- e-map-∃ f (sels e e₁) = _ , f (sels (proj₁ (e-map-∃ f e)) (proj₁ (e-map-∃ f e₁)))
  -- e-map-∃ f (sel e e₁) = _ , f (sel (proj₁ (e-map-∃ f e)) (proj₁ (e-map-∃ f e₁)))
  -- e-map-∃ f (selb x e e₁) = _ , f (selb x (proj₁ (e-map-∃ f e)) (proj₁ (e-map-∃ f e₁)))
  -- e-map-∃ f (zero-but e e₁ e₂) = _ , f (zero-but (proj₁ (e-map-∃ f e)) (proj₁ (e-map-∃ f e₁)) (proj₁ (e-map-∃ f e₂)))
  -- e-map-∃ f (slide e x e₁ x₁) = _ , f (slide (proj₁ (e-map-∃ f e)) x (proj₁ (e-map-∃ f e₁)) x₁)
  -- e-map-∃ f (backslide e e₁ x x₁) = _ , f (backslide (proj₁ (e-map-∃ f e)) (proj₁ (e-map-∃ f e₁)) x x₁)
  -- e-map-∃ f (bin x e e₁) = _ , f (bin x (proj₁ (e-map-∃ f e)) (proj₁ (e-map-∃ f e₁)))
  -- e-map-∃ f (scaledown x e) = _ , f (scaledown x (proj₁ (e-map-∃ f e)))
  -- e-map-∃ f (let′ e e₁) = _ , f (let′ (proj₁ (e-map-∃ f e)) (proj₁ (e-map-∃ f e₁)))
  -- e-map-∃ f (un x e) = _ , f (un x (proj₁ (e-map-∃ f e)))
  -- e-map-∃ f e = _ , f e

  -- e-map-∃ : {Γ : Ctx} {A : ∀ {Δ ip} → E Δ ip → Set}
  --   → (∀ {Δ ip} → (e : E Δ ip) → ∃ (λ (x : E Δ ip) → A x))
  --   → (e : E Γ is) → ∃ (λ (x : E Γ is) → A x)
  -- e-map-∃ f (imaps e) = f (imaps (proj₁ (f (proj₁ (e-map-∃ f e)))))
  -- e-map-∃ f (imap e) = f (imap (proj₁ (f (proj₁ (e-map-∃ f e)))))
  -- e-map-∃ f (imapb x e) = f (imapb x (proj₁ (f (proj₁ (e-map-∃ f e)))))
  -- e-map-∃ f (sum e) = f (sum (proj₁ (f (proj₁ (e-map-∃ f e)))))
  -- e-map-∃ f (maximum e) = f (maximum (proj₁ (f (proj₁ (e-map-∃ f e)))))
  -- e-map-∃ f (sels e e₁) = f (sels (proj₁ (f (proj₁ (e-map-∃ f e)))) (proj₁ (f (proj₁ (e-map-∃ f e₁)))))
  -- e-map-∃ f (sel e e₁) = f (sel (proj₁ (f (proj₁ (e-map-∃ f e)))) (proj₁ (f (proj₁ (e-map-∃ f e₁)))))
  -- e-map-∃ f (selb x e e₁) = f (selb x (proj₁ (f (proj₁ (e-map-∃ f e)))) (proj₁ (f (proj₁ (e-map-∃ f e₁)))))
  -- e-map-∃ f (zero-but e e₁ e₂) = f (zero-but (proj₁ (f (proj₁ (e-map-∃ f e)))) (proj₁ (f (proj₁ (e-map-∃ f e₁)))) (proj₁ (f (proj₁ (e-map-∃ f e₂)))))
  -- e-map-∃ f (slide e x e₁ x₁) = f (slide (proj₁ (f (proj₁ (e-map-∃ f e)))) x (proj₁ (f (proj₁ (e-map-∃ f e₁)))) x₁)
  -- e-map-∃ f (backslide e e₁ x x₁) = f (backslide (proj₁ (f (proj₁ (e-map-∃ f e)))) (proj₁ (f (proj₁ (e-map-∃ f e₁)))) x x₁)
  -- e-map-∃ f (bin x e e₁) = f (bin x (proj₁ (f (proj₁ (e-map-∃ f e)))) (proj₁ (f (proj₁ (e-map-∃ f e₁)))))
  -- e-map-∃ f (scaledown x e) = f (scaledown x (proj₁ (f (proj₁ (e-map-∃ f e)))))
  -- e-map-∃ f (let′ e e₁) = f (let′ (proj₁ (f (proj₁ (e-map-∃ f e)))) (proj₁ (f (proj₁ (e-map-∃ f e₁)))))
  -- e-map-∃ f (un x e) = f (un x (proj₁ (f (proj₁ (e-map-∃ f e)))))
  -- e-map-∃ f e = f e

  -- e-map-∃ : {Γ : Ctx} {P : ∀ {Δ ip} → E Δ ip → Set}
  --   → (∀ {Δ ip} → (e : E Δ ip) → ∃ (λ (x : E Δ ip) → P x))
  --   → (e : E Γ is) → ∃ (λ (x : E Γ is) → P x)
  -- e-map-∃ f e = {!   !} , {!   !}

  -- e-elim : {P : Ctx → IS → Set}
  --   → (∀ {is} → P ε is )
  --   → (∀ Γ ip {is} → P Γ is → P (Γ ▹ ip) is)
  --   → (∀ Γ {is} → P Γ is)
  -- e-elim base ind ε = base
  -- e-elim base ind (Γ ▹ ip) = ind Γ ip (e-elim base ind Γ)

  -- e-elim : {P : ∀ {Γ is} → E Γ is → Set}
  --   → (∀ {Γ s} → P {Γ} {ar s} zero) --zero
  --   → (∀ {Γ s} → P {Γ} {ar s} one) --one
  --   → (∀ {Γ is} (v : is ∈ Γ) → P {Γ } {is} (var v)) --var
  --   → (∀ {Γ s} (e : E (Γ ▹ ix s) (ar unit))  → P (imaps e)) --imaps
  --   → (∀ {Γ s} (e : E (Γ ▹ ix s) (ar p))  → P (imap e)) --imap
  --   → (∀ {Γ s p q} (x : s * p ≈ q) (e : E (Γ ▹ ix s) (ar p))  → P (imapb x e)) --imapb
  --   → (∀ {Γ s p q} (e : E Γ (ar s)) (i : E Γ (ix s))  → P (imapb x e)) --sels
  --   -- → (∀ {Γ} → (f : ) → (e : ) → P _ _ (f e)) --zero-but
  --   -- → (∀ {Γ} → (f : ) → (e : ) → P _ _ (f e)) --slide
  --   -- → (∀ {Γ} → (f : ) → (e : ) → P _ _ (f e)) --backslide
  -- e-elim = {!   !}

  -- e-map : {A : Ctx → IS → Set} → (∀ {Δ ip} → E Δ ip → A Δ ip) → E Γ is → A Γ is
  -- e-map f e = proj₂ (e-map-∃ (λ x → x , (f x)) e)

  -- e-map-⊆ : (∀ {Δ ip} → E Δ ip → Γ ⊆ Δ → E Δ ip) → E Γ is → E Γ is
  -- e-map-⊆ {is = is} f e = (e-map f e) ⊆-eq

  -- count-sels : E Γ is → ip ∈ Γ → ℕ
  -- count-sels e v = e-map count-sels' e v 0 where
  --   count-sels' : E Γ is → ip ∈ Γ → ℕ → ℕ
  --   count-sels' (sels e (var i)) v count with eq? i v
  --   ... | veq = (count-sels' e v count) + 1
  --   ... | neq .i y = count
  --   count-sels' (sel e (var i)) v count with eq? i v
  --   ... | veq = (count-sels' e v count) + 1
  --   ... | neq .i y = count
  --   count-sels' (selb x e (var i)) v count with eq? i v
  --   ... | veq = (count-sels' e v count) + 1
  --   ... | neq .i y = count
  --   count-sels' e v count = count

  -- test-count : ℕ
  -- test-count = count-sels {Γ = ε ▹ ix unit} {is = ar unit} (sels one (var v₀) ⊞ sels one (var v₀)) v₀
  -- -- WkSub.test-count

  -- inline : E Γ is → E Γ is
  -- inline e = norm-lets (e-map inline' e) where
  --   inline' : E Γ is → E Γ is
  --   inline' (let′ e e₁) with a ← (inline' e₁) | count-uses a v₀ | e
  --   ... | 0 | _ = sub a (sub-id ▹ (inline' e))
  --   ... | _ | var b = sub a (sub-id ▹ (var b))
  --   ... | _ | zero = sub a (sub-id ▹ zero)
  --   ... | _ | one = sub a (sub-id ▹ one)
  --   -- ... | 1 | (imap b) = {!   !}
  --   -- ... | 1 | _ = sub a (sub-id ▹ (inline' e))
  --   ... | _ | _ = let′ (inline' e) a
  --   inline' e = e

  -- e-map₂ : (∀ {Δ ip iq} → Γ ⊆ Δ → E Δ ip → E Δ iq → E Δ ip)
  --        → (∀ {Δ ip iq} → Γ ⊆ Δ → E Δ iq → E Δ ip → E Δ ip)
  --        → E Γ is → E Γ ir → E Γ is
  -- e-map₂ f g a b =
  --   f ⊆-eq (e-map (λ s x → f s x (wk s b)) a)
  --          (e-map (λ s x → g s (wk s a) x) b)

  -- e-mapr₂ : (∀ {Δ ip iq} → Γ ⊆ Δ → E Δ ip → E Δ iq → E Δ ip)
  --        → E Γ is → E Γ ir → E Γ is
  -- e-mapr₂ f a b = e-map₂ f (λ s x y → f s y x) a b

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

  Max : ∀ {Γ}
       → (GE (Γ ▹ ix s) (ix s) → E (Γ ▹ ix s) (ar p))
       → E Γ (ar p)
  Max f = maximum (f λ {Δ} ⦃ p ⦄ → var (V v₀))

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

    tiles : ∀ {Γ} → E Γ (ar []) → E Γ (ar s)
    tiles x = Imaps (λ _ → ⟨ x ⟩)

    tile : ∀ {Γ} → E Γ (ar s) → E Γ (ar (p ⊗ s))
    tile x = Imap (λ _ → ⟨ x ⟩)

    iswap : ∀ {Γ} → E Γ (ar (s ⊗ u)) → E Γ (ar (u ⊗ s))
    iswap {s} {u} x = Imap {u} λ i → Imaps {s} λ j → sels (sel ⟨ x ⟩ j) i

    iswap3 : ∀ {Γ} → E Γ (ar (p ⊗ (s ⊗ u))) → E Γ (ar (s ⊗ (p ⊗ u)))
    iswap3 {p} {s} {u} x = Imap {s} λ i → Imap {p} λ j → sel (sel ⟨ x ⟩ j) i

    linear : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar s) → E Γ (ar u)
    linear {u} {s} w x =
      Imaps {u} λ i → Sum {s} λ j → sels (sel ⟨ w ⟩ i ⊠ ⟨ x ⟩) j
      -- Imaps {u} λ i → Sum {s} λ j → sels (sel ⟨ w ⟩ i) j ⊠ sels ⟨ x ⟩ j

    m-linear : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (p ⊗ s)) → E Γ (ar (p ⊗ u))
    m-linear {u} {s} {p} w xs = Imap {p} λ i → linear ⟨ w ⟩ (sel ⟨ xs ⟩ i)

    matmult : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (r ⊗ s)) → E Γ (ar (u ⊗ r))
    matmult {u} {s} {r} w1 w2 =
      Imap {u} λ i → Imaps {r} λ j → Sum {s} λ k →
      sels ((sel ⟨ w1 ⟩ i) ⊠ (sel ⟨ w2 ⟩ j)) k

    matmul : ∀ {Γ} → E Γ (ar (u ⊗ s)) → E Γ (ar (s ⊗ r)) → E Γ (ar (u ⊗ r))
    matmul {u} {s} {r} w1 w2 =
      Imap {u} λ i → Imaps {r} λ j → Sum {s} λ k →
      sels (sel ⟨ w1 ⟩ i) k ⊠ sels (sel ⟨ w2 ⟩ k) j

    -- stabilize : ∀ {Γ} → E Γ (ar s) → E Γ (ar s)
    -- stabilize {s = s} x = x ⊟ tiles (Max {s} (λ i → sels ⟨ x ⟩ i))

    -- Is this correct?
    softmax : ∀ {Γ} → E Γ (ar s) → E Γ (ar s)
    softmax {s = s} x =
      -- Let exps := 𝕖^ x In
      -- Let total := Sum {s} (λ i → sels exps i) In
      -- exps ⊠ (tiles $ 𝟙/ total)
      Let maxs := Max {s} (λ i → sels ⟨ x ⟩ i) In
      Let exps := 𝕖^ (⟨ x ⟩ ⊟ tiles maxs) In
      Let total := Sum {s} (λ i → sels exps i) In
      exps ⊠ (tiles $ 𝟙/ total)

    -- m-stabilize : ∀ {Γ} → E Γ (ar (s ⊗ p)) → E Γ (ar (s ⊗ p))
    -- m-stabilize {s = s} x = Imap {s} λ i → stabilize (sel ⟨ x ⟩ i)

    m-softmax : ∀ {Γ} → E Γ (ar (s ⊗ p)) → E Γ (ar (s ⊗ p))
    m-softmax {s} {p} {Γ} x =
      Imap {s} λ i → (softmax (sel ⟨ x ⟩ i))

      -- Let maxs := Imap {s} (λ i → Imaps {p} (λ j → Max (λ k → sels (sel ⟨ x ⟩ i) k))) In
      -- Imap {s} λ i → (softmax (sel ⟨ x ⟩ i ⊟ sel maxs i))

      -- Imap {s} λ i → (
      --   Let xi := sel ⟨ x ⟩ i In
      --   Let maxs := Max (λ j → sels xi j) In
      --   softmax (xi ⊟ (tiles maxs)))

    rmsnorm : ∀ {Γ} → E Γ (ar s) → E Γ (ar s)
    rmsnorm {s = s} x =
      Let xx := x ⊠ x In
      Let ms := scaledown (len s) (Sum (λ i → sels xx i)) In
      Let scale := sqrt (ms ⊞ (scaledown 100000 one)) In
      Imaps λ i → (sels ⟨ x ⟩ i) // scale

    m-rmsnorm : ∀ {Γ} → E Γ (ar (s ⊗ p)) → E Γ (ar (s ⊗ p))
    m-rmsnorm {s} {p} {Γ} x = Imap {s} λ i → rmsnorm (sel ⟨ x ⟩ i)

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

    attention : ∀ {Γ} (sc : ℕ)
                   (mask : E Γ (ar (sl ⊗ sl)))
                   (qs ks vs : E Γ (ar (sl ⊗ hd)))
                  → E Γ (ar (sl ⊗ hd))
    attention {sl} {hd} {Γ} sc mask hqs hks hvs =
      -- Let hqks := matmult {sl} hqs hks In
      -- Let masked := (scaledown sc hqks) ⊞ ⟨ mask ⟩ In
      -- Let maxs := Imap {sl} (λ i → tiles (Max (λ j → sels (sel masked i) j))) In
      -- Let sf := m-softmax {sl} (masked ⊟ maxs) In
      -- matmul {sl} sf ⟨ hvs ⟩
      Let hqks := matmult {sl} hqs hks In
      Let masked := (scaledown sc hqks) ⊞ ⟨ mask ⟩ In
      Let sf := m-softmax {sl} (masked) In
      matmul {sl} sf ⟨ hvs ⟩

    mh-attention : ∀ {Γ} (sc : ℕ)
                   (mask : E Γ (ar (sl ⊗ sl)))
                   (qs ks vs : E Γ (ar (ah ⊗ (sl ⊗ hd))))
                  → E Γ (ar (ah ⊗ (sl ⊗ hd)))
    mh-attention {sl} {ah} {hd} {Γ} sc mask bqs bks bvs =
      Imap {ah} λ i →
      attention {sl} sc ⟨ mask ⟩ (sel ⟨ bqs ⟩ i) (sel ⟨ bks ⟩ i) (sel ⟨ bvs ⟩ i)

    block-tok : ∀ {Γ} → E Γ (ar ed) → ah * hd ≈ ed → E Γ (ar (ah ⊗ hd))
    block-tok {ed} {ah} {hd} {Γ} x pr = Imap {ah} λ i → selb pr ⟨ x ⟩ i

    unblock-tok : ∀ {Γ} → E Γ (ar (ah ⊗ hd)) → ah * hd ≈ ed → E Γ (ar ed)
    unblock-tok {ah} {hd} {ed} {Γ} x pr = Imapb pr λ i → sel ⟨ x ⟩ i

    block-vec : ∀ {Γ} → E Γ (ar (sl ⊗ ed)) → ah * hd ≈ ed
                → E Γ (ar (ah ⊗ (sl ⊗ hd)))
    block-vec {sl} {ed} {ah} x pr =
      iswap3 {sl} {ah} (Imap {sl} λ i → block-tok (sel ⟨ x ⟩ i) pr)

    unblock-vec : ∀ {Γ} → E Γ (ar (ah ⊗ (sl ⊗ hd))) → ah * hd ≈ ed
      → E Γ (ar (sl ⊗ ed))
    unblock-vec {ah} {sl} {hd} {ed} {Γ} x pr = Imap {sl} λ i →
      unblock-tok (sel (iswap3 {ah} {sl} ⟨ x ⟩) i) pr

    mgpt-forward : ∀ {ah hd : S} {Γ} (sc : ℕ) (mask : E Γ (ar (sl ⊗ sl)))
                   (p : GPT-Params Γ vo ed sl fd) (wseq : E Γ (ar (sl ⊗ ed)))
                   → ah * hd ≈ ed → E Γ (ar (sl ⊗ vo))
    mgpt-forward {sl} {vo} {ed} {fd} {ah} {hd} {Γ} sc mask p wseq pr =
      Let wpe-wseq := (p .wpe) ⊞ wseq In
      Let seq := m-rmsnorm {sl} wpe-wseq In
      -- layer pass
      Let nseq := m-rmsnorm {sl} seq In
        -- attention block
      Let qs := m-linear {u = ed} ⟨ p .wqry ⟩ nseq In
      Let ks := m-linear {u = ed} ⟨ p .wkey ⟩ nseq In
      Let vs := m-linear {u = ed} ⟨ p .wval ⟩ nseq In
      Let bqs := block-vec qs pr In
      Let bks := block-vec ks pr In
      Let bvs := block-vec vs pr In
      Let battn := mh-attention {sl} {ah} {hd} sc ⟨ mask ⟩ bqs bks bvs In
      Let attn := unblock-vec battn pr In
      Let oseq := m-linear {u = ed} {p = sl} ⟨ p .wout ⟩ attn In
      Let cseq := oseq ⊞ seq In
        -- mlp block
      Let nseq2 := m-rmsnorm {sl} cseq In
      Let useq := m-linear {p = sl} ⟨ p .wup ⟩ nseq2 In
      Let aseq := relu useq In
      Let dseq := m-linear {u = ed} {p = sl} ⟨ p .wdown ⟩ aseq In
      Let lseq := dseq ⊞ cseq In
      -- build logits
      --Let logits := m-linear {u = vo} {p = sl} ⟨ p .wvoc ⟩ lseq In logits
      m-linear {u = vo} {p = sl} ⟨ p .wvoc ⟩ lseq

    cross-entropy : ∀ {Γ} (logits target : E Γ (ar s)) → (E Γ (ar []))
    cross-entropy {s} logits target =
      -- Let lnsf := ln (softmax (stabilize logits)) In
      Let lnsf := ln (softmax (logits)) In
      (⊟ (Sum λ i → sels lnsf i ⊠ sels ⟨ target ⟩ i))
      -- (⊟ (Sum λ i → sels (ln (softmax ⟨ logits ⟩)) i ⊠ sels ⟨ target ⟩ i))

    m-cross-entropy : ∀ {Γ} (logits target : E Γ (ar (s ⊗ p))) → (E Γ (ar s))
    m-cross-entropy {s} {p} logits target =
      Imaps λ i → cross-entropy {p} (sel ⟨ logits ⟩ i) (sel ⟨ target ⟩ i)
      -- Let m-sf := Imap {s} (λ i → softmax (sel ⟨ logits ⟩ i)) In
      -- Let m-ln := ln m-sf In
      -- Let m-mul := m-ln ⊠ ⟨ target ⟩ In
      -- Let m-s := Imaps (λ i → Sum (λ j → sels (sel m-mul i) j)) In m-s

    -- mgpt-loss : ∀ {ah hd : S} {Γ} (sc : ℕ) (mask : E Γ (ar (sl ⊗ sl)))
    --                (p : GPT-Params Γ vo ed sl fd) (wseq : E Γ (ar (sl ⊗ ed)))
    --                (target : E Γ (ar (sl ⊗ vo))) → ah * hd ≈ ed → E Γ (ar [])
    -- mgpt-loss {sl} {vo} {ed} {fd} {ah} {hd} {Γ} sc mask p wseq target pr =
    --   Let logits := mgpt-forward sc mask p wseq pr In
    --   Let losses := m-cross-entropy {sl} logits ⟨ target ⟩ In
    --   Let loss := avg losses In loss

    mgpt-loss : ∀ {ah hd : S} {Γ} (sc : ℕ) (mask : E Γ (ar (sl ⊗ sl)))
                   (p : GPT-Params Γ vo ed sl fd) (wseq : E Γ (ar (sl ⊗ ed)))
                   (target : E Γ (ar (sl ⊗ vo))) → ah * hd ≈ ed → E Γ (ar [])
    mgpt-loss {sl} {vo} {ed} {fd} {ah} {hd} {Γ} sc mask p wseq target pr =

      Let wpe-wseq := (p .wpe) ⊞ wseq In
      Let seq := m-rmsnorm {sl} wpe-wseq In
      -- layer pass
      Let nseq := m-rmsnorm {sl} seq In
        -- attention block
      Let qs := m-linear {u = ed} ⟨ p .wqry ⟩ nseq In
      Let ks := m-linear {u = ed} ⟨ p .wkey ⟩ nseq In
      Let vs := m-linear {u = ed} ⟨ p .wval ⟩ nseq In
      Let bqs := block-vec qs pr In
      Let bks := block-vec ks pr In
      Let bvs := block-vec vs pr In
      Let battn := mh-attention {sl} {ah} {hd} sc ⟨ mask ⟩ bqs bks bvs In
      Let attn := unblock-vec battn pr In
      Let oseq := m-linear {u = ed} {p = sl} ⟨ p .wout ⟩ attn In
      Let cseq := oseq ⊞ seq In
        -- mlp block
      Let nseq2 := m-rmsnorm {sl} cseq In
      Let useq := m-linear {p = sl} ⟨ p .wup ⟩ nseq2 In
      Let aseq := relu useq In
      Let dseq := m-linear {u = ed} {p = sl} ⟨ p .wdown ⟩ aseq In
      Let lseq := dseq ⊞ cseq In
      -- build logits
      Let logits := m-linear {u = vo} {p = sl} ⟨ p .wvoc ⟩ lseq In
      -- calculate losses
      Let losses := m-cross-entropy {sl} logits ⟨ target ⟩ In
      -- average loss
      --Let loss := avg losses In loss
      avg losses

    ED = ι 16 ; AH = ι 4 ; HD = ι 4 ; SL = ι 16 ; FD = ι 64 ; SC = 2 ; VO = ι 27

    PR : AH * HD ≈ ED
    PR = cons

    rmsnorm-e : E _ _
    rmsnorm-e = Lcon (ar (ι 5 ⊗ ι 6) ∷ []) (ar (ι 5 ⊗ ι 6)) ε (λ x → rmsnorm {s = ι 5 ⊗ ι 6} x)

    div-e : E _ _
    div-e = Lcon (ar (ι 6) ∷ ar (ι 6) ∷ []) (ar (ι 6)) ε (λ x y → (x ⊞ y) // (x ⊞ y))

    softmax-e : E _ _
    softmax-e = Lcon (ar (ι 2) ∷ ar (ι 2) ∷ []) (ar (ι 2)) ε (λ i x → softmax {s = ι 2} x)

    softmax-inline : ∀ {Γ} → E Γ (ar s) → E Γ (ar s)
    softmax-inline {s = s} x = (𝕖^ x) // tiles (Sum {s} (λ i → sels (𝕖^ ⟨ x ⟩) i))

    softmax-inline-e : E _ _
    softmax-inline-e = Lcon (ar (ι 5 ⊗ ι 6) ∷ []) (ar (ι 5 ⊗ ι 6)) ε (λ x → softmax-inline {s = ι 5 ⊗ ι 6} x)

    test : ∀ {Γ} → E Γ (ar s) → E Γ (ar s)
    test {s = s} x = softmax x
      -- -- tiles $ Sum {s} (λ i → sels ⟨ x ⟩ i)
      -- Let total := Sum {s} (λ i → sels ⟨ x ⟩ i) In
      -- Let r := ⟨ x ⟩ ⊠ tiles total In
      -- Sum (λ i → sels r i)

    test-e : E _ _
    test-e = Lcon (ar (ι 5) ∷ ar (ι 5) ∷ []) (ar (ι 5)) ε (λ s x → test {s = ι 5} x)

    id-e : E _ _
    id-e = Lcon (ar (ι 5 ⊗ ι 6) ∷ []) (ar (ι 5 ⊗ ι 6)) ε (λ x → x)

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

    -- let-test : ∀ {Γ} → E Γ (ar SL) → E Γ (ar [])
    -- let-test x =
    --   Let a := (Let b := Imaps (λ i → sels ⟨ x ⟩ i) In scaledown 2 b) In Sum (λ i → sel a i)
