---
title: "Correctness Meets Performance"
author: "<u>Artjoms Šinkarovs</u>, Troels Henriksen"
date: "14 October 2025"
institute: "University of Southampton"

controls: false
lang: "en-GB"
theme: "white"
margin: 0.1
showNotes: "false" 
...

# Motivation

- Numerical code is frequently written in unsafe languages that rely
  on unverified libraries: C, Fortran, Python, MKL, OpenBLAS, etc.
    
- Correct‑by‑construction specifications rarely run efficiently:
  Lean, Rocq, Agda, etc.

# Vision

Rather than pursuing a single, idealised language, we adopt a pragmatic
compromise: combine a dependently‑typed proof assistant with
a high‑performance back‑end.

## 

<img src="img/agdafut-04.jpg" style="max-width: 90%; max-height: 90%;" />

# In the Paper

We implement a canonical CNN in Agda, including AD required for training;
we extract the specification and run it in Futhark.

## Technical Contributions

- Rank-polymorphic array theory, combinators, CNN.

- $E$ --- a deeply-embedded DSL:
  - HOAS-style wrappers for $E$ and combinators;
  - Automatic differentiation $E \to E$;
  - Semantically-preserving optimiser $E \to E$.

- Extraction from $E$ to Futhark.

- Experimental evaluation.



# Running Example

CNN for handwritten digit recognition, trained on the MNIST data set.


## Manual Implementation

In the array langauge SaC:

```c
float [10,1,1,1,1] 
forward (float [28,28] I, float [6,5,5] k1,
         float [6] b1, float [12,6,5,5] k2,
         float [12] b2, float [10,12,1,4,4] k3,
         float [10] b) {
  c1 = logisitc (mconv (I, k1, b1));
  s1 = avgpool (c1);
  c2 = logisitc (mconv (s1, k2, b2 ));
  s2 = avgpool (c2);
  return logisitc (mconv (s2, k3, b));
}
```

# Array Theory

Key idea: represent arrays as functions from indices to values and define
rank‑polymorphic combinators.


## Arrays
<!--
```
open import Data.Nat using (zero; suc; ℕ)
open import Relation.Binary.PropositionalEquality
open import Relation.Nullary
open import Data.List using (List; []; _∷_)
open import Data.Empty
open import Function
open import Data.Fin as F using (zero; suc; Fin; combine; remQuot; fromℕ<; inject+; splitAt)
open import Data.Product as Prod hiding (map; zipWith) -- using (∃; ∃[_]_; _,_; _×_; uncurry)

postulate
 ⋯ : ∀ {A : Set} → A

module _ where
module A where
 open import Data.Nat using (zero; suc; ℕ; _+_; _*_; _≤_; s≤s; z≤n; _<_)
 open import Data.Nat.Properties using (+-mono-≤; ≤-step; ≤-pred; _≟_; +-comm; +-suc)
```
-->

```
 data S : Set where
   []   : S
   _∷_  : ℕ → S → S
```
<!--
```
 variable
   m n k : ℕ
   s p q r u w : S
   X Y Z : Set
```
-->
```
 data P : S → Set where
   []   : P []
   _∷_  : Fin n → P s → P (n ∷ s)
```
```
 Ar : S → Set → Set
 Ar s X = P s → X
```

. . .

As a container: `Ar = List ℕ ◃ All Fin`

## Combinators (1)

```
 K : X → Ar s X
 K x i = x
 
 map : (X → Y) → Ar s X → Ar s Y
 map f a i = f (a i)
 
 zipWith : (X → Y → Z) → Ar s X → Ar s Y → Ar s Z
 zipWith f a b i = f (a i) (b i)
```

## Combinators (2)

```
 _⊗_ : S → S → S
 _⊗ₚ_ : P s → P p → P (s ⊗ p)
 split : P (s ⊗ p) → P s × P p
```
<!--
```
 [] ⊗ p = p
 (n ∷ s) ⊗ p = n ∷ (s ⊗ p)
 
 [] ⊗ₚ jv = jv
 (i ∷ iv) ⊗ₚ jv = i ∷ (iv ⊗ₚ jv)
 
 split {s = []}    is = [] , is
 split {s = x ∷ s} (i ∷ is) = Prod.map₁ (i ∷_) (split is)
 
 _≟ₚ_ : (i j : P s) → Dec (i ≡ j)
 _≟ₚ_ {[]} [] [] = yes refl
 _≟ₚ_ {x ∷ s} (i ∷ is) (j ∷ js) with i F.≟ j
 ... | no ¬p = no λ { refl → ¬p refl }
 ... | yes refl with is ≟ₚ js
 ... | no ¬q = no λ { refl → ¬q refl }
 ... | yes refl = yes refl
```
-->
```
 nest : Ar (s ⊗ p) X → Ar s (Ar p X)
 nest a i j = a (i ⊗ₚ j)
 
 unnest : Ar s (Ar p X) → Ar (s ⊗ p) X
 unnest a i = uncurry a (split i)
```

<!--
```
 pattern ι n = n ∷ []
 
 ιsuc : P (ι n) → P (ι (suc n))
 ιsuc (ι i) = ι (suc i)
 
 sum₁ : (X → X → X) → X → Ar (ι n) X → X
 sum₁ {n = zero}   f ε a = ε
 sum₁ {n = suc n}  f ε a = f (a (ι zero)) (sum₁ f ε (a ∘ ιsuc))
 
 sum : (X → X → X) → X → Ar s X → X
 sum {s = []}     f ε a = a []
 sum {s = x ∷ s}  f ε a = sum₁ f ε $ map (sum f ε) (nest a)
```
-->


## Convolution (1-d)

<!--
```
 Vec : ℕ → Set → Set;      Ix : ℕ → Set   
 postulate
```
```
  _⊕_ : Fin m → Fin (1 + n) → Fin (m + n)
  _⊝_ : (i : Fin (m + n)) (j : Fin m)
      → Dec (∃[ k ] j ⊕ k ≡ i)
 ℝ = ℕ
```
. . .
-->


```
 Vec m X = Ar (ι m) X;     Ix m = P (ι m)

 slide₁ : Ix m → Vec (m + n) X → Vec (1 + n) X
 slide₁ (ι i) v (ι j) = v (ι (i ⊕ j))
```

. . .

```
 conv₁ : Vec (m + n) ℝ → Vec m ℝ → Vec (1 + n) ℝ
 conv₁ a w = sum₁ (zipWith _+_) (K 0)
                  (λ i → map (w i *_)
                             (slide₁ i a))
```

## Convolution (n-d)

<!--
```
 data Pw₂ (R : (a b : ℕ) → Set) 
      : (a b : S) → Set where instance
     []    : Pw₂ R [] []
     cons  : ⦃ R m n ⦄ → ⦃ Pw₂ R s p ⦄
           → Pw₂ R (m ∷ s) (n ∷ p) 
 data Pw₃ (R : (a b c : ℕ) → Set) 
      : (a b c : S) → Set where instance
     []    : Pw₃ R [] [] []
     cons  : ⦃ R m n k ⦄ → ⦃ Pw₃ R s p q ⦄
           → Pw₃ R (m ∷ s) (n ∷ p) (k ∷ q)

 infix 5 _+_≈_
 infix 5 suc_≈_
 infix 5 _*_≈_
 _+_≈_ : (s p q : S) → Set
 _+_≈_ = Pw₃ (_≡_ ∘₂ _+_)
 _*_≈_ : (s p q : S) → Set
 _*_≈_ = Pw₃ (_≡_ ∘₂ _*_)
 suc_≈_ : (s p : S) → Set
 suc_≈_ = Pw₂ (_≡_ ∘ suc)

 postulate
  backslide : P s → Ar u X → suc p ≈ u → (def : X)
            → s + p ≈ r → Ar r X
```

-->
```
  slide : P s → s + p ≈ r → Ar r X 
        → suc p ≈ u → Ar u X
```

```
 conv : s + p ≈ r → Ar r ℝ → Ar s ℝ 
      → suc p ≈ u → Ar u ℝ
 conv sp a w su = sum (zipWith _+_) (K 0) 
                      (λ i → map (w i *_)
                                 (slide i sp a su))

```

## CNN

<!--
```
 postulate
   avgp₂ : (m n : ℕ) → Ar (m * 2 ∷ n * 2 ∷ []) ℝ → Ar (m ∷ n ∷ []) ℝ
   logistic : Ar s ℝ → Ar s ℝ
 
 mconv : ⦃ s + p ≈ r ⦄ → Ar r ℝ → Ar (u ⊗ s) ℝ → Ar u ℝ → ⦃ suc p ≈ q ⦄ → Ar (u ⊗ q) ℝ
 mconv ⦃ sp ⦄ inp w b ⦃ su ⦄ = unnest λ i → map (b i +_) (conv sp inp (nest w i) su)
 
 forward : (inp  :  Ar (28 ∷ 28 ∷ []) ℝ) → (k₁ : Ar (6 ∷ 5 ∷ 5 ∷ []) ℝ)
         → (b₁   :  Ar (6  ∷ []) ℝ)      → (k₂ : Ar (12 ∷ 6 ∷ 5 ∷ 5 ∷ []) ℝ)
         → (b₂   :  Ar (12 ∷ []) ℝ)      → (fc : Ar (10 ∷ 12 ∷ 1 ∷ 4 ∷ 4 ∷ []) ℝ)
         → (b    :  Ar (10 ∷ []) ℝ)      → Ar (10 ∷ 1 ∷ 1 ∷ 1 ∷ 1 ∷ []) ℝ
```
-->
```
 forward inp k₁ b₁ k₂ b₂ fc b = let
     c₁ = logistic $ mconv inp k₁ b₁ 
     
     s₁ = unnest {s = 6 ∷ []} 
          $ map (avgp₂ 12 12) (nest c₁)
     
     c₂ = logistic $ mconv s₁ k₂ b₂ 
     
     s₂ = unnest {s = 12 ∷ 1 ∷ []}
          $ map (avgp₂ 4 4) (nest c₂)
     
     r = logistic $ mconv s₂ fc b 
   in r
```

# DSL

We now define an intrinsically typed (shaped) embedded DSL, $E$,
that captures the fundamental array combinators.


## Types and Contexts
<!--
```
module Lang where
 open A hiding (sum; slide; backslide; logistic)
 infixl 15 _▹_
```
-->
```
 data IS : Set where
   ix  : S → IS
   ar  : S → IS
```

. . .

```
 data Ctx : Set where
   ε    : Ctx
   _▹_  : Ctx → IS → Ctx
```
<!--
```
 variable
   Γ Δ Ξ Ψ : Ctx
   is ip iq ir : IS
```
-->
```
 data _∈_ : IS → Ctx → Set where
   v₀  : is ∈ (Γ ▹ is)
   vₛ  : is ∈ Γ → is ∈ (Γ ▹ ip)
```

## Definition of E

<div style="font-size:60%;">
```
 data E : Ctx → IS → Set where
   var        : is ∈ Γ → E Γ is
   zero       : E Γ (ar s)
   one        : E Γ (ar s)
 
   imaps      : E (Γ ▹ ix s) (ar []) → E Γ (ar s)
   sels       : E Γ (ar s) → E Γ (ix s) → E Γ (ar [])
 
   imap       : E (Γ ▹ ix s) (ar p) → E Γ (ar (s ⊗ p))
   sel        : E Γ (ar (s ⊗ p)) → E Γ (ix s) → E Γ (ar p)
 
   imapb      : s * p ≈ q → E (Γ ▹ ix s) (ar p) → E Γ (ar q)
   selb       : s * p ≈ q → E Γ (ar q) → E Γ (ix s) → E Γ (ar p)
 
   sum        : E (Γ ▹ ix s) (ar p) → E Γ (ar p)
   zero-but   : E Γ (ix s) → E Γ (ix s) → E Γ (ar p) → E Γ (ar p)
 
   slide      : E Γ (ix s) → s + p ≈ r → E Γ (ar r) → suc p ≈ u → E Γ (ar u)
   backslide  : E Γ (ix s) → E Γ (ar u) → suc p ≈ u → s + p ≈ r → E Γ (ar r)
 
   logistic   : E Γ (ar s) → E Γ (ar s)
   _⊞_ _⊠_    : E Γ (ar s) → E Γ (ar s) → E Γ (ar s)
   scaledown  : ℕ → E Γ (ar s) → E Γ (ar s)
   minus      : E Γ (ar s) → E Γ (ar s)
   let′       : E Γ (ar s) → E (Γ ▹ ar s) (ar p) → E Γ (ar p)
 
 pattern _⊟_ a b = a ⊞ (minus b)
```
</div>

<!--
```
 infixl 10 _⊞_
 infixl 15 _⊠_

```
-->

## Evaluation (1)


```
record Real : Set₁ where
 field
   R : Set
   fromℕ : ℕ → R
   _+_ _*_ _÷_ : R → R → R
   -_ e^_ : R → R
```
<!--
```
 infixl 10 _+_ 
 infixl 15 _*_ 
 infixl 15 _÷_ 
```
-->
```
 logisticʳ : R → R
 logisticʳ x = fromℕ 1 ÷ (fromℕ 1 + e^ (- x))

 0ᵣ = fromℕ 0
 1ᵣ = fromℕ 1
```

## Evaluation (2)

<!--
```
module Eval (r : Real) where
 open import Data.Unit
 open Lang
 open Real r
 open A

```
-->
```
 Val : IS → Set
 Val (ar s)  = Ar s R
 Val (ix s)  = P s

 Env : Ctx → Set
 Env ε         = ⊤
 Env (Γ ▹ is)  = Env Γ × Val is

 ⟦_⟧ : E Γ is → ⦃ Env Γ ⦄ → Val is
```
<!--
```
 ⟦_⟧ = ⋯

module WkSub where
  open Lang

  data _⊆_ : Ctx → Ctx → Set where
    ε     : ε ⊆ ε
    skip  : Γ ⊆ Δ → Γ ⊆ (Δ ▹ is)
    keep  : Γ ⊆ Δ → (Γ ▹ is) ⊆ (Δ ▹ is)
  
  wkv : Γ ⊆ Δ → is ∈ Γ → is ∈ Δ
  wkv (skip s) v       = vₛ (wkv s v)
  wkv (keep s) v₀      = v₀
  wkv (keep s) (vₛ v)  = vₛ (wkv s v)
  
  wk : Γ ⊆ Δ → E Γ is → E Δ is
  
  ⊆-eq : Γ ⊆ Γ
  ⊆-eq {ε}      = ε
  ⊆-eq {Γ ▹ x}  = keep ⊆-eq
  
  _↑ : E Γ is → E (Γ ▹ ip) is
  _↑ = wk (skip ⊆-eq)
  
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
  wk s (logistic e) = logistic (wk s e)
  wk s (e ⊞ e₁) = (wk s e) ⊞ (wk s e₁)
  wk s (e ⊠ e₁) = (wk s e) ⊠ (wk s e₁)
  wk s (scaledown x e) = scaledown x (wk s e)
  wk s (minus e) = minus (wk s e)
  wk s (let′ e e₁) = let′ (wk s e) (wk (keep s) e₁)
  
  data Sub (Γ : Ctx) : Ctx → Set where
    ε    : Sub Γ ε
    _▹_  : Sub Γ Δ → E Γ is → Sub Γ (Δ ▹ is)
  
  wks : Sub Γ Δ → Γ ⊆ Ψ → Sub Ψ Δ
  wks ε p        = ε
  wks (s ▹ x) p  = (wks s p) ▹ wk p x
  
  sdrop : Sub Γ Δ → Sub (Γ ▹ is) Δ
  sdrop s = wks s (skip ⊆-eq)

  skeep : Sub Γ Δ → Sub (Γ ▹ is) (Δ ▹ is)
  skeep s = sdrop s ▹ var v₀
  sub-id : Sub Γ Γ
  sub-id {ε}      = ε
  sub-id {Γ ▹ x}  = skeep sub-id

  sub : E Δ is → Sub Γ Δ → E Γ is
  subv : Sub Γ Δ → is ∈ Δ → E Γ is
  subv (s ▹ x) v₀      = x
  subv (s ▹ x) (vₛ v)  = subv s v
  
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
  sub (logistic e) s = logistic (sub e s)
  sub (e ⊞ e₁) s = (sub e s) ⊞ (sub e₁ s)
  sub (e ⊠ e₁) s = (sub e s) ⊠ (sub e₁ s)
  sub (scaledown x e) s = scaledown x (sub e s)
  sub (minus e) s = minus (sub e s)
  sub (let′ e e₁) s = let′ (sub e s) (sub e₁ (skeep s))

  _∙ˢ_ : Sub Δ Ψ → Sub Γ Δ → Sub Γ Ψ
  ε ∙ˢ t = ε
  (s ▹ x) ∙ˢ t = (s ∙ˢ t) ▹ sub x t
  sub-swap : Sub (Γ ▹ is ▹ ip) (Γ ▹ ip ▹ is)
  sub-swap = sdrop (sdrop sub-id) ▹ var v₀ ▹ var (vₛ v₀)

```
-->


<!-- Here we include the module with hoas slides -->

!include build/hoas.md


# Automatic Differentiation

Reverse mode automatic differentiation for the intrinsically typed language.


## Environment

<!--
```
module AD where
  open import Data.Unit
  open import Data.Product as Prod
  open A hiding (sum; backslide; slide; logistic)
  open WkSub
  open Lang
  --open Syntax
```
-->
```
  data Env : Ctx → Ctx → Set where
    ε     : Env ε Γ
    skip  : Env Γ Δ → Env (Γ ▹ ix s) Δ
    _▹_   : Env Γ Δ → E Δ (ar s)
          → Env (Γ ▹ ar s) Δ

  data EE : Ctx → Ctx → Set where
    env   : Env Γ Δ → EE Γ Δ
    let′  : E Δ (ar s) → EE Γ (Δ ▹ ar s)
          → EE Γ Δ 
```

## AD

<!--
```
  ee-update+  : EE Γ Δ → ar s ∈ Γ → E Δ (ar s) → EE Γ Δ
  _▹𝟘         : EE Γ Δ → EE (Γ ▹ ar s) (Δ ▹ ar s)
  ∇ₗ : E Γ (ar s) → EE (Γ ▹ ar s) Γ → EE Γ Γ
  ∇Σ : (e s : E (Γ ▹ ix s) (ar p)) → EE Γ Γ → EE Γ Γ
  ee-update+ = ⋯
  _▹𝟘 = ⋯
  ∇ₗ = ⋯
  ∇Σ = ⋯
```
-->

<div style="font-size:60%;">
```
  ∇ : (e s : E Γ is) → EE Γ Γ → EE Γ Γ
  ∇ {is = ix _} (var x)    s   = id
  ∇ {is = ar _} (var x)    s   = λ δ → ee-update+ δ x s
  ∇ zero                   s   = id
  ∇ one                    s   = id

  ∇ (imaps e)              s   = ∇Σ e (sels     (s ↑) (var v₀))
  ∇ (imap e)               s   = ∇Σ e (sel      (s ↑) (var v₀))
  ∇ (E.imapb m e)          s   = ∇Σ e (E.selb m (s ↑) (var v₀))

  ∇ (sels e i)             s   = ∇ e (imaps     (zero-but (var v₀) (i ↑) (s ↑)))
  ∇ (sel e i)              s   = ∇ e (imap      (zero-but (var v₀) (i ↑) (s ↑)))
  ∇ (E.selb m e i)         s   = ∇ e (E.imapb m (zero-but (var v₀) (i ↑) (s ↑)))

  ∇ (E.sum e)              s   = ∇Σ e (s ↑) 
  ∇ (zero-but i j e)       s   = ∇ e (zero-but i j s) 

  ∇ (E.slide i p e su)     s   = ∇ e (E.backslide i s su p) 
  ∇ (E.backslide i e su p) s   = ∇ e (E.slide i p s su) 

  ∇ (e ⊞ e₁)               s   = ∇ e s ∘ ∇ e₁ s
  ∇ (e ⊠ e₁)               s   = ∇ e (s ⊠ e₁) ∘ ∇ e₁ (s ⊠ e)
  ∇ (scaledown x e)        s   = ∇ e (scaledown x s)
  ∇ (minus e)              s   = ∇ e (minus s)
  ∇ (logistic e)           s   = ∇ e (let′ (logistic e) 
                                     ((s ↑) ⊠ var v₀ ⊠ (one ⊟ var v₀)))
  
  ∇ (let′ e e₁)            s   = λ δ → ∇ₗ e (let′ e (∇ e₁ (s ↑) (δ ▹𝟘)))
```
</div>

# Optimiser and Codegen

Before code generation we apply a suite of semantically preserving
rewrite rules.


## Optimiser

<!--
```
module Opt where
  open import Data.Nat as ℕ using (ℕ; zero; suc)
  open import Data.Product
  open Lang
  open WkSub
```
-->
```
  record RealProp (r : Real) : Set where
    open Real r; field
      +-neutˡ : ∀ {x} → 0ᵣ + x  ≡ x
      +-neutʳ : ∀ {x} → x + 0ᵣ  ≡ x
      *-neutˡ : ∀ {x} → 1ᵣ * x  ≡ x
      *-neutʳ : ∀ {x} → x * 1ᵣ  ≡ x
```
<!--
```
  postulate
    real : Real

  open Eval real
```
-->
```
  _≈ᵛ_ : (a b : Val is) → Set
  _≈ᵛ_ {ix s} a b = a ≡ b
  _≈ᵛ_ {ar s} a b = ∀ i → a i ≡ b i

  _≈ᵉ_ : E Γ is → E Γ is → Set
  _≈ᵉ_ {Γ} a b = ⦃ ρ : Env Γ ⦄ → ⟦ a ⟧ ≈ᵛ ⟦ b ⟧
  
  opt : (e : E Γ is) → ∃[ e′ ] (e ≈ᵉ e′)
```
<!--
```
  opt = ⋯
```
-->

## Codegen

<!--
```
module Futhark where
  open import Data.Unit
  open import Data.String
  open A hiding (Ix)
  open Lang
  
  open import Effect.Monad.State
  open import Effect.Monad using (RawMonad)
  open RawMonadState {{...}} -- public
  open RawMonad {{...}} -- public
  
  instance
    _ = monad
    _ = applicative
    _ = monadState
```
-->
```
  data Ix : S → Set where 
    []  : Ix []
    _∷_ : String → Ix s → Ix (n ∷ s)

  Sem : IS → Set
  Sem (ar s) = Ix s → State ℕ ((String → String)
                               × String)
  Sem (ix s) = Ix s

  FEnv : Ctx → Set
  FEnv ε = ⊤
  FEnv (Γ ▹ is) = FEnv Γ × Sem is

  to-fut : E Γ is → FEnv Γ → State ℕ (Sem is)
  to-str : E Γ (ar s) → FEnv Γ → State ℕ String
```
<!--
```
  to-fut = ⋯
  to-str = ⋯
```
-->

# Performance

Training times for various problem sizes, comparing Futhark with TensorFlow on
an NVIDIA A100 GPU.


<table>
  <tr>
    <th>Size</th>
    <th>Futhark</th>
    <th>TensorFlow</th>
    <th>Ratio</th>
  </tr>
  <tr>
    <td>$10000$</td>
    <td>$0.91s$</td>
    <td>$1.07s$</td>
    <td>$0.85\times{}$</td>
  </tr>
  <tr>
    <td>$60000$</td>
    <td>$4.93s$</td>
    <td>$2.92s$</td>
    <td>$1.68\times{}$</td>
  </tr>
</table>

<!--
# Conclusions

The proposed collaboration between theorem provers and high‑performance
languages is viable in practice.

* Correctness guarantees:
  - Absence of out-of-bound indexing;
  - Certain functions are proven to be inverses;
  - Well-scopedness and well-typedness of the embedded DSL;
  - Semantics-preserving optimisations.

* Efficient execution on GPUs.
-->

##

<img src="img/agdafutlove.jpg" style="max-width: 90%; max-height: 90%;" />


# Thank you!

I am hiring PhD students in Southampton 
[a.sinkarovs@soton.ac.uk](mailto:a.sinkarovs@soton.ac.uk).




