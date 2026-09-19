{-# OPTIONS --guardedness #-}
--{-# OPTIONS --warn=noUserWarning #-}
module ToFile where

open import Extraction
open Extract
open import Data.String
open import Lang
open import IO

main : Main
-- main = run (putStrLn Extract.mgpt-forward-s)
main = run (putStrLn Extract.mgpt-loss-s)
-- main = run (putStrLn grad-mgpt-loss-s)
-- main = run (putStrLn grad-mgpt-loss-pp)