{-# OPTIONS --guardedness #-}
--{-# OPTIONS --warn=noUserWarning #-}
module ToFile where

open import Extraction
open Extract
open import Data.String
open import Lang
open import IO

main : Main
-- main = run (putStrLn Extract.model-pp)
-- main = run (putStrLn Extract.model-s)
-- main = run (putStrLn Extract.gpt-loss-s)
-- main = run (putStrLn Extract.gpt-loss-flat-s)
main = run (putStrLn Extract.grad-gpt-loss-flat-s)
-- main = run (putStrLn Extract.grad-gpt-loss-flat-pp)
-- main = run (putStrLn Extract.grad-gpt-loss-s)
-- main = run (putStrLn Extract.grad-gpt-loss-pp)