{-# OPTIONS --warn=noUserWarning #-}
module jairo.experiments where
  open import Data.Nat as ℕ using (ℕ)
  open import Data.Bool as B using (if_then_else_)
  {-
  open import Data.Float as F using (_+_; _*_; _÷_; e^_; -_; fromℕ; sqrt; _≤ᵇ_)
    renaming (Float to R)
  -}
  open import Data.List as L using (List; []; _∷_; reverse)
  open import Data.List.Properties
  open import Data.List.Relation.Unary.All as All using (All; []; _∷_)
  open import Data.Fin as Fin using (zero; suc; Fin; combine; remQuot; fromℕ<;
    inject+; splitAt)
  import Relation.Binary.PropositionalEquality as Eq
  open Eq using (_≡_; refl; trans; sym; cong; cong-app; subst)
  open Eq.≡-Reasoning using (begin_; step-≡; _∎)
  open import Function
  open import Data.Product as Prod using (∃; _,_; _×_; proj₁; proj₂)

  infix 4 _⊡_
  _⊡_ = trans

  open import Ar

  record JReal : Set₁ where
    field
      JR : Set
      fromℕ : ℕ → JR
      _+_ _*_ _÷_ _∧_ : JR → JR → JR
      -_ e^_ sqrt_ : JR → JR

    infixl 10 _+_
    infixl 15 _*_
    infixl 15 _÷_
    infixl 15 _∧_

    0ᵣ : JR
    0ᵣ = fromℕ 0

  module microgpt (jr : JReal) where
    open JReal jr

    {- matrix-vector multiplication -}
    linear : Ar (u ⊗ s) JR → Ar s JR → Ar u JR
    linear w x i = sum _+_ 0ᵣ (zipWith _*_ (nest w i) x)

    {- A transpose? -}
    swap : Ar (u ⊗ s) JR → Ar (s ⊗ u) JR
    swap {u} {s} x = unnest {s} λ i j → nest x j i

    {- matrix multiplication -}
    matmul : Ar (u ⊗ s) JR → Ar (s ⊗ r) JR → Ar (u ⊗ r) JR
    matmul {u} {s} {r} w x = unnest {u} (λ i j → linear w (λ k → nest x k j) i)

    {- converts a vector into a probability distribution
      In microgpt they substract the max value of the vector to
      prevent overflow of exp. Should we do the same?
    -}
    softmax : Ar s JR → Ar s JR
    softmax {s} x  = let
      exps : Ar s JR
      exps = map e^_ x

      total : JR
      total = sum _+_ 0ᵣ exps

      r = map (_÷ total) exps
      in r

    {- number of data in an array
      Not sure if this is how I should calculate the length of an array-}
    lenS : S → ℕ
    lenS s = L.foldl ℕ._*_ 1 s

    {- rename to size -}
    len : Ar s JR → ℕ
    len {s} _ = lenS s

    {- rescales a vector so its values have unit root-mean-square.
      They add 0.0001 in one of the steps. I assume to avoid
      overflow. Should we do the same?
      -}
    rmsnorm : Ar s JR → Ar s JR
    rmsnorm {s} x = let
      scale : JR
      scale = sqrt ((sum _+_ 0ᵣ (zipWith _*_ x x)) ÷ (fromℕ (len x)))

      r = map (_* scale) x
      in r

    {- returns the maximum between two floats.
        Maybe use Dec instead of Bool. -}

    {- rectified linear unit -}
    relu : Ar s JR → Ar s JR
    relu = map (0ᵣ ∧_)

    {- returns the reverse of a shape.
      I make my own implementation because the og uses fold
      and idk how to define reverseP for that shape. -}
    reverseS : S → S
    reverseS [] = []
    reverseS (x ∷ xs) = (reverseS xs) ⊗ (x ∷ [])

    {- returns the reverse of a position based onf reverseS -}
    reverseP : P s → P (reverseS s)
    reverseP {[]} [] = []
    reverseP {s ∷ ss} (x ∷ xs) = reverseP xs ++ (x ∷ [])

    {- from reverse to the original.
       I need this to define transpose -}
    unreverseP : P (reverseS s) → P s
    unreverseP {[]} [] = []
    unreverseP {s ∷ ss} x = x₁ ++ x₂ where
      x₁ : P (s ∷ [])
      x₁ = proj₂ (splitP x)

      x₂ : P ss
      x₂ = unreverseP (proj₁ (splitP x))

    {- Transpose of an array? -}
    transpose : Ar s JR → Ar (reverseS s) JR
    transpose x i = x (unreverseP i)

    {- computes an attention block -}
    attention : {u s r t : S} → Ar (u ⊗ s) JR → Ar (r ⊗ s) JR → Ar (r ⊗ t) JR
              → Ar (u ⊗ t) JR
    attention {u} {s} {r} {t} q k v = let
      {- Are len and swap correct here? -}
      scale : JR
      scale = (sqrt (fromℕ (len v)))

      l : Ar (u ⊗ r) JR
      l = map (_÷ scale) (matmul {u} q (swap {r} k))

      w : Ar (u ⊗ t) JR
      w = matmul {u} (softmax l) v
      in w

    {- Multi-headed attention-}
    mattention : {h u s r t : S}
               → Ar (h ⊗ (u ⊗ s)) JR → Ar (h ⊗ (r ⊗ s)) JR → Ar (h ⊗ (r ⊗ t)) JR
               → Ar (h ⊗ (u ⊗ t)) JR
    mattention {h} {u} q k v =
      unnest {h} λ i → attention {u} (nest q i) (nest k i) (nest v i)

    {- A single layer forward pass. -}
    gptLayer :
        {ah hd sl fd : S} →
        let is = (ah ⊗ (hd ⊗ sl)) in
          (inp : Ar is JR)
        → (wa : Ar (4 ∷ [] ⊗ (is ⊗ is)) JR)
        → (wf : Ar (2 ∷ [] ⊗ ((fd ⊗ is) ⊗ is)) JR)
        → Ar is JR
    gptLayer {ah} {hd} {sl} {fd} inp wa wf = let
      is = ah ⊗ (hd ⊗ sl)

      ninp = rmsnorm inp

      q : Ar is JR
      q = linear (nest wa (zero ∷ [])) ninp

      k : Ar is JR
      k = linear (nest wa (suc zero ∷ [])) ninp

      v : Ar is JR
      v = linear (nest wa (suc (suc zero) ∷ [])) ninp

      wo : Ar (is ⊗ is) JR
      wo = nest wa (suc (suc (suc zero)) ∷ [])

      wf1 : Ar ((fd ⊗ is) ⊗ is) JR
      wf1 = nest wf (zero ∷ [])

      wf2 : Ar (is ⊗ (fd ⊗ is)) JR
      wf2 = swap {fd ⊗ is} (nest wf (suc zero ∷ []))

      c₁ : Ar is JR
      c₁ = mattention {ah} {hd} q k v

      s₁ : Ar is JR
      s₁ = zipWith _+_ (linear wo c₁) inp

      s₂ : Ar (fd ⊗ is) JR
      s₂ = relu $ linear wf1 (rmsnorm s₁)

      c₃ : Ar is JR
      c₃ = linear wf2 s₂

      r = zipWith _+_ c₃ s₁
      in r

    sucP : P (n ∷ [] ⊗ s) → P (ℕ.suc n ∷ [] ⊗ s)
    sucP (i ∷ is) = suc i ∷ is

    tail : Ar (ℕ.suc n ∷ [] ⊗ s) X → Ar (n ∷ [] ⊗ s) X
    tail x = x ∘ sucP

    head : Ar (ℕ.suc n ∷ [] ⊗ s) X → Ar s X
    head x is = x (zero ∷ is)

    {-
    GPT2 with the following simplifications:
      1. RMSNorm instead of LayerNorm
      2. no biases
      3. ReLU instaed of GeLU
    1. 2. could be potential be included. 3. is more difficult since
    it uses a probability density function.
      -}
    gpt : {n : ℕ} {ah hd sl fd : S} →
        let is = (ah ⊗ (hd ⊗ sl)) in
          Ar is JR
        → Ar (n ∷ [] ⊗ (4 ∷ [] ⊗ (is ⊗ is))) JR
        → Ar (n ∷ [] ⊗ (2 ∷ [] ⊗ ((fd ⊗ is) ⊗ is))) JR
        → Ar is JR
    gpt {n} {ah} {hd} {sl} {fd} inp wa wf =
      sum’ (λ w' inp' → gptLayer {ah} {hd} {fd = fd} inp' (proj₁ w') (proj₂ w'))
        inp λ (i : P (n ∷ [])) → Prod._,_ (nest wa i) (nest wf i)

    {- embedding dimension (C) -}
    ED : ℕ
    ED = 16

    {-
    number of attention heads (H)
    Must be such that ED/AH is a natural
    -}
    AH : ℕ
    AH = 4

    {- number of layers (N)
      It is 1 microgpt so perhaps we do not need it? -}
    NL : ℕ
    NL = 1

    {- Dimension of each head -}
    HD : ℕ
    HD = ED ℕ./ AH

    {- sequence length (T)
       A sequence is a list of tokens.
       In microgpt tokens are letters and sequences names -}
    SL : ℕ
    SL = 16

    {- The feed forward network projects into FDx-}
    FD : ℕ
    FD = 4

    {- I chose this order to make it easy to pass into mattention-}
    IS : S
    IS = (AH ∷ []) ⊗  ((HD ∷ []) ⊗ (SL ∷ []))

    agpt :
          (inp : Ar IS JR)
        → Ar (NL ∷ [] ⊗ (4 ∷ [] ⊗ (IS ⊗ IS))) JR
        → Ar (NL ∷ [] ⊗ (2 ∷ [] ⊗((FD ∷ [] ⊗ IS) ⊗ IS))) JR
        → Ar IS JR
    agpt = gpt {NL} {AH ∷ []} {HD ∷ []} {SL ∷ []} {FD ∷ []}
