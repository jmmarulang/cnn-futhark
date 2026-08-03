{-# OPTIONS --backtracking-instance-search #-} -- only needed for tests
--{-# OPTIONS --warn=noUserWarning #-}
module _ where

open import Grad

module Optimise where
  open import Lang
  open import Data.Product
  open import Data.Nat
  -- We are not interested in the proof,
  -- but we are interested in the optimisation, so we ignore
  -- the R module, and make it up with a postulate

  open import Real
  postulate
    r : Real.Real
    rp : RealProp r

  open import Opt r rp public

  doopt : E Γ is → E Γ is
  doopt e = opt e .proj₁

  multiopt : E Γ is → ℕ → E Γ is
  multiopt e 0 = e
  multiopt e (suc n) = doopt (multiopt e n)

module Extract where
  open import Data.String
  open import Text.Printf
  open import Data.Product as Prod
  open import Data.Nat using (ℕ; zero; suc; _+_)
  open import Data.List as L
  open import Relation.Binary.PropositionalEquality

  open import Lang
  open import Ar hiding (r)
  open import Function
  open import Futhark
  open import Replace

  open import Effect.Monad.State
  open import Effect.Monad using (RawMonad)
  open RawMonadState {{...}} --public
  open RawMonad {{...}} --public

  instance
    _ = monad
    _ = applicative
    _ = monadState

  open Optimise
  open Syntax
  open Primitives
  open WkSub

  OPT = 20

  -- Show Env (e.g. after running grad) where optimisations are applied
  -- to every expression in the list.
  env-opt : Env Γ Δ → Env Γ Δ
  env-opt ε = ε
  env-opt (skip ρ) = skip (env-opt ρ)
  env-opt (ρ ▹ x) = env-opt ρ ▹ multiopt x OPT

  ee-opt : EE Γ Δ → EE Γ Δ
  ee-opt (env ρ) = env (env-opt ρ)
  ee-opt (let′ x ρ) = let′ (multiopt x OPT) (ee-opt ρ)

  env-count-uses : Env Γ Δ → is ∈ Δ → ℕ
  env-count-uses ε v = 0
  env-count-uses (skip ρ) v = env-count-uses ρ v
  env-count-uses (ρ ▹ x) v = env-count-uses ρ v + count-uses x v

  ee-count-uses : EE Γ Δ → is ∈ Δ → ℕ
  ee-count-uses (env ρ) = env-count-uses ρ
  ee-count-uses (let′ x ρ) v = count-uses x v + ee-count-uses ρ (there v)

  env-norm-lets : Env Γ Δ → Env Γ Δ
  env-norm-lets ε = ε
  env-norm-lets (skip x) = skip (env-norm-lets x)
  env-norm-lets (xs ▹ x) = env-norm-lets xs ▹ norm-lets x

  ee-inline : EE Γ Δ → EE Γ Δ
  ee-inline (env x) = env (env-norm-lets x)
  ee-inline (let′ x ρ) with δ ← ee-inline ρ | ee-count-uses δ v₀
  ... | 0 = ee-sub δ (sub-id ▹ norm-lets x) -- does nothing?
  ... | 1 = ee-sub δ (sub-id ▹ norm-lets x) -- why only for 1?
  ... | _ = let′ (norm-lets x) δ

  ee-inline' : EE Γ Δ → EE Γ Δ
  ee-inline' (env x) = env x
  ee-inline' (let′ x ρ) with δ ← ee-inline' ρ | ee-count-uses δ v₀
  ... | 0 = ee-sub δ (sub-id ▹ x) -- does nothing?
  ... | 1 = ee-sub δ (sub-id ▹ x) -- why only for 1?
  ... | _ = let′ x δ

  env-replace : Env Γ Δ → (a b : E Δ is) → Env Γ Δ
  env-replace ε a b = ε
  env-replace (skip ρ) a b = skip (env-replace ρ a b)
  env-replace (ρ ▹ x) a b = env-replace ρ a b ▹ replace x a b

  ee-replace : EE Γ Δ → (a b : E Δ is) → EE Γ Δ
  ee-replace (env ρ) x y = env (env-replace ρ x y)
  ee-replace (let′ e e₁) x y = let′ (replace e x y) (ee-replace e₁ (x ↑) (y ↑))

  ee-dedup : EE Γ Δ → EE Γ Δ
  ee-dedup (env x) = env x
  ee-dedup (let′ x e) = let′ x (ee-replace (ee-dedup e) (x ↑) (var v₀))

  ee-OPT : EE Γ Δ → EE Γ Δ
  ee-OPT ρ = ee-inline $ ee-opt (ee-inline (ee-opt $ ee-inline ρ)) --??

  data NamedEnv : Ctx → Set where
    ε : NamedEnv ε
    _▹_ : NamedEnv Γ → String → NamedEnv (Γ ▹ is)

  from-named : NamedEnv Γ → FEnv Γ
  from-named ε = _
  from-named (_▹_ {is = ix s} ρ x) = from-named ρ , fresh-ix x
  from-named (_▹_ {is = ar s} ρ x) = from-named ρ , mkar x

  -- Show chain using SemFuthark
  env-fut′ : Env Γ Δ → NamedEnv Γ → NamedEnv Δ → State ℕ String
  env-fut′ ε ρ ν = return ""
  env-fut′ (skip e) (ρ ▹ _) ν = env-fut′ e ρ ν
  env-fut′ (e ▹ x) (ρ ▹ n) ν = do
    r ← env-fut′ e ρ ν
    v ← to-str x (from-named ν)
    return $ printf "%s\nlet d%s = %s" r n v

  ee-fut′ : EE Γ Δ → NamedEnv Γ → NamedEnv Δ → State ℕ String
  ee-fut′ (env ρ) = env-fut′ ρ
  ee-fut′ (let′ {s = s} x e) ρ ν = do
    c ← get
    modify suc
    v ← to-str x (from-named ν)
    let n = fresh-var c
    r ← ee-fut′ e ρ (ν ▹ n)
    return $ printf "let %s = %s\n%s" n v r

  -- Apply optimisations and generate the code.
  ee-fut : EE Γ Γ → NamedEnv Γ → String
  ee-fut e ρ = proj₂ $ runState (ee-fut′ (ee-OPT $ ee-dedup $ ee-OPT e) ρ ρ) 0

  -- nodedup-ee-fut : EE Γ Γ → NamedEnv Γ → String
  -- nodedup-ee-fut e ρ = proj₂ $ runState (ee-fut′ (ee-opt $ ee-opt e) ρ ρ) 0

  -- This is the "entry point" that computes derivatives
  -- and generates the Futhark code for the variable names
  -- passed through NamedEnv
  pp : E Γ (ar s) → NamedEnv Γ → String
  pp e ρ = ee-fut ({- env-norm-lets $ -} grad e one zero-ee) ρ

  -- nodedup-pp : E Γ (ar s) → NamedEnv Γ → String
  -- nodedup-pp e ρ = nodedup-ee-fut ({- env-norm-lets $ -} grad e one zero-ee) ρ

  -- Examples
  -- ========
  conv-e : E _ _
  conv-e = Lcon (ar (5 ∷ 5 ∷ []) ∷ ar (2 ∷ 2 ∷ []) ∷ []) (ar (4 ∷ 4 ∷ [])) ε
           λ img k1 → Let t := Primitives.Cnn.conv img k1 In
                      logi t -- wrapped inside a logistic?

  grad-conv-e = pp conv-e (ε ▹ "img" ▹ "k1")
  {-
  "let x0 = (imap2 4 4 (\\ x2_0 x2_1 -> (isum2 2 2 (\\ x1_0 x1_1 -> (img[(x1_0 + x2_0)][(x1_1 + x2_1)] F.* k1[x1_0][x1_1])))))
  let x3 = (let x4 = (imap2 4 4 (\\ x6_0 x6_1 -> (logistics x0[x6_0][x6_1])))
  in (imap2 4 4 (\\ x5_0 x5_1 -> (x4[x5_0][x5_1] F.* (one F.+ (F.neg x4[x5_0][x5_1]))))))

  let dimg = (imap2 5 5 (\\ x8_0 x8_1 -> (isum2 2 2 (\\ x7_0 x7_1 -> if (x8_0 >= x7_0 && x8_1 >= x7_1 && (x8_0 - x7_0) < 4 && (x8_1 - x7_1) < 4) then (x3[(x8_0 - x7_0)][(x8_1 - x7_1)] F.* k1[x7_0][x7_1]) else zero))))
  let dk1 = (imap2 2 2 (\\ x9_0 x9_1 -> (isum2 4 4 (\\ x10_0 x10_1 -> (x3[x10_0][x10_1] F.* img[(x9_0 + x10_0)][(x9_1 + x10_1)])))))"
  -}

  grad-conv-s = pp conv-e (ε ▹ "inp" ▹ "k1") -- whats the difference?

  {-
  "let x0 = (imap2 4 4 (\\ x2_0 x2_1 -> (isum2 2 2 (\\ x1_0 x1_1 -> (inp[(x1_0 + x2_0)][(x1_1 + x2_1)] F.* k1[x1_0][x1_1])))))
  let x3 = (let x4 = (imap2 4 4 (\\ x6_0 x6_1 -> (logistics x0[x6_0][x6_1])))
  in (imap2 4 4 (\\ x5_0 x5_1 -> (x4[x5_0][x5_1] F.* (one F.+ (F.neg x4[x5_0][x5_1]))))))

  let dinp = (imap2 5 5 (\\ x8_0 x8_1 -> (isum2 2 2 (\\ x7_0 x7_1 -> if (x8_0 >= x7_0 && x8_1 >= x7_1 && (x8_0 - x7_0) < 4 && (x8_1 - x7_1) < 4) then (x3[(x8_0 - x7_0)][(x8_1 - x7_1)] F.* k1[x7_0][x7_1]) else zero))))
  let dk1 = (imap2 2 2 (\\ x9_0 x9_1 -> (isum2 4 4 (\\ x10_0 x10_1 -> (x3[x10_0][x10_1] F.* inp[(x9_0 + x10_0)][(x9_1 + x10_1)])))))"
  -}

  compc1 : E _ _
  compc1 =  Lcon (  ar (28 ∷ 28 ∷ []) ∷ ar (6 ∷ 5 ∷ 5 ∷ [])
                  ∷ ar (6 ∷ []) ∷ ar (12 ∷ 6 ∷ 5 ∷ 5 ∷ [])
                  ∷ ar (12 ∷ []) ∷ [])

                  --(ar (12 ∷ 1 ∷ 8 ∷ 8 ∷ [])) ε
                  (ar (12 ∷ 1 ∷ 8 ∷ 8 ∷ [])) ε
            λ inp k₁ b₁ k₂ b₂ →
            Let c₁₁ := Primitives.Cnn.mconv inp k₁ b₁  In
            Let c₁ := logi c₁₁ In
            Let s₁  := (Imap {s = 6 ∷ []} λ i → Primitives.Cnn.avgp₂ 12 12 (sel c₁ i)) In
            Let c₂₁ := Primitives.Cnn.mconv s₁ k₂ b₂ In
            c₂₁

  grad-compc1-e = ee-opt (grad compc1 one zero-ee)
  grad-compc1-s = pp compc1 (ε ▹ "inp" ▹ "k1" ▹ "b1" ▹ "k2" ▹ "b2")

  -- test-e : E _ _
  -- test-e = Lcon (ar ([]) ∷  ar [] ∷ []) (ar ([])) ε
  --          λ a x  → (a ⊞ a) ⊠ x
  -- test-s = pp test-e (ε ▹ "a" ▹ "x")

  -- test-n = WkSub.norm-lets test-e

  sum-let : E _ _
  sum-let = Lcon (ar (5 ∷ []) ∷ ar (5 ∷ []) ∷ []) (ar []) ε
            λ a b → Sum λ i → (Let x := sels a i ⊞ sels b i In x ⊠ x)
  sum-let-s = pp sum-let (ε ▹ "a" ▹ "b")

  -- grad-test-e = ee-opt (grad test-e (var v₀) zero-ee)
  -- grad-test-s = ee-fut (grad test-e (var v₀) zero-ee) (ε ▹ "x" ▹ "s" )

  grad-cnn-e = ee-OPT (grad Primitives.Cnn.cnn one zero-ee)

  -- This is our CNN example from the paper.
  grad-cnn-s = pp Primitives.Cnn.cnn (ε ▹ "inp" ▹ "k1" ▹ "b1" ▹ "k2" ▹ "b2" ▹ "fc" ▹ "b" ▹ "target" )

  -- Jairo made
  -- grad-avg-ee : EE (ε ▹ ar Microgpt.SL) (ε ▹ ar Microgpt.SL)
  -- grad-avg-ee = ee-opt $ ee-dedup $ grad Primitives.Microgpt.avg-e one zero-ee

  -- grad-avg-s : String
  -- grad-avg-s = pp Primitives.Microgpt.avg-e (ε ▹ "inp")

  -- grad-test-sels-s : String
  -- grad-test-sels-s = pp Primitives.Microgpt.test-sels-e (ε ▹ "inp")

  -- grad-test-let-ee : EE _ _
  -- grad-test-let-ee = ee-opt $ ee-dedup $ grad Primitives.Microgpt.test-let-e one zero-ee

  -- grad-test2-let-s : String
  -- grad-test2-let-s = pp Primitives.Microgpt.test2-let ε

  -- grad-test-let-s : String
  -- grad-test-let-s = pp Primitives.Microgpt.test-let-e (ε ▹ "inp")

  -- grad-test3-let-s : String
  -- grad-test3-let-s = pp Primitives.Microgpt.test3-let-e (ε ▹ "inp")

  -- -- grad-nodedup-test-let-s : String
  -- -- grad-nodedup-test-let-s = nodedup-pp Primitives.Microgpt.test-let-e (ε ▹ "inp")

  -- cross-entropy-s : String
  -- cross-entropy-s = proj₂ (runState (to-str Primitives.Microgpt.cross-entropy-e (from-named (ε ▹ "inp" ▹ "target"))) 0)

  -- grad-cross-entropy-s : String
  -- grad-cross-entropy-s = pp Primitives.Microgpt.cross-entropy-e (ε ▹ "inp" ▹ "target")

  -- m-softmax-s : String
  -- m-softmax-s = proj₂ (runState (to-str (multiopt Primitives.Microgpt.m-softmax-e OPT) (from-named (ε ▹ "inp"))) 0)

  -- sel-zb-s : String
  -- sel-zb-s = proj₂ (runState (to-str (multiopt Primitives.Microgpt.sel-zb-e OPT) (from-named (ε ▹ "inp"))) 0)

  -- x64-s : String
  -- x64-s = proj₂ (runState (to-str (multiopt Primitives.Microgpt.x64-e OPT) (from-named (ε ▹ "inp"))) 0)

  -- imap-imapb-sum-zerobut-s : String
  -- imap-imapb-sum-zerobut-s = proj₂ (runState (to-str (multiopt Primitives.Microgpt.imap-imapb-sum-zerobut-e OPT) (from-named (ε ▹ "inp"))) 0)

  -- sum-imap-imapb-zerobut-s : String
  -- sum-imap-imapb-zerobut-s = proj₂ (runState (to-str (multiopt Primitives.Microgpt.sum-imap-imapb-zerobut-e OPT) (from-named (ε ▹ "inp"))) 0)

  -- unblock-tok-s : String
  -- unblock-tok-s = proj₂ (runState (to-str (multiopt Primitives.Microgpt.unblock-tok-e OPT) (from-named (ε ▹ "inp"))) 0)

  -- let-test-s : String
  -- let-test-s = proj₂ (runState (to-str (multiopt Primitives.Microgpt.let-test-e OPT) (from-named (ε ▹ "inp"))) 0)

  grad-mgpt-loss-e = ee-OPT $ ee-dedup $ ee-OPT (grad Primitives.Microgpt.mgpt-loss-e one zero-ee)

  mgpt-loss-s : String
  mgpt-loss-s = proj₂ (runState (to-str (multiopt Primitives.Microgpt.mgpt-loss-e OPT) ((from-named (ε ▹ "mask" ▹ "wpe" ▹ "wqry" ▹ "wkey" ▹ "wval" ▹ "wout" ▹ "wup" ▹ "wdown" ▹ "wvoc" ▹ "wseq" ▹ "target")))) 0)

  -- mgpt-loss-s : String
  -- mgpt-loss-s = proj₂ (runState (to-str Primitives.Microgpt.mgpt-loss-e (from-named (ε ▹ "mask" ▹ "wpe" ▹ "wqry" ▹ "wkey" ▹ "wval" ▹ "wout" ▹ "wup" ▹ "wdown" ▹ "wvoc" ▹ "wseq" ▹ "target"))) 0)

  mgpt-forward-s : String
  mgpt-forward-s = proj₂ (runState (to-str ( multiopt Primitives.Microgpt.mgpt-forward-e OPT) (from-named (ε ▹ "mask" ▹ "wpe" ▹ "wqry" ▹ "wkey" ▹ "wval" ▹ "wout" ▹ "wup" ▹ "wdown" ▹ "wvoc" ▹ "wseq"))) 0)

  grad-mgpt-loss-s : String
  grad-mgpt-loss-s = pp Primitives.Microgpt.mgpt-loss-e
    (ε ▹ "mask" ▹ "wpe" ▹ "wqry" ▹ "wkey" ▹ "wval" ▹ "wout" ▹ "wup"
       ▹ "wdown" ▹ "wvoc" ▹ "wseq" ▹ "target")