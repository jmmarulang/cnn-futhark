{-# OPTIONS --guardedness #-}
--{-# OPTIONS --warn=noUserWarning #-}
module ToFile where


open import Extraction
open Extract
open import Data.String
open import Lang
open import IO

grad-mgpt-loss-s : String
grad-mgpt-loss-s = pp Primitives.Microgpt.mgpt-loss-e
    (ε ▹ "mask" ▹ "wpe" ▹ "wqry" ▹ "wkey" ▹ "wval" ▹ "wout" ▹ "wup"
       ▹ "wdown" ▹ "wvoc" ▹ "wseq" ▹ "target")

main : Main
-- main = run (putStrLn Extract.grad-avg-s)
-- main = run (putStrLn Extract.mgpt-loss-s)
-- main = run (putStrLn grad-mgpt-loss-s)
-- main = run (putStrLn Extract.grad-cnn-s)