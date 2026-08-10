{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE EmptyCase #-}
{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -Wno-overlapping-patterns #-}

module MAlonzo.Code.LangEq where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.Bool
import qualified MAlonzo.Code.Agda.Builtin.Equality
import qualified MAlonzo.Code.Agda.Builtin.List
import qualified MAlonzo.Code.Agda.Builtin.Maybe
import qualified MAlonzo.Code.Agda.Builtin.Sigma
import qualified MAlonzo.Code.Ar
import qualified MAlonzo.Code.Data.Irrelevant
import qualified MAlonzo.Code.Data.List.Base
import qualified MAlonzo.Code.Data.Maybe.Base
import qualified MAlonzo.Code.Data.Nat.Properties
import qualified MAlonzo.Code.Lang
import qualified MAlonzo.Code.Relation.Nullary.Decidable.Core
import qualified MAlonzo.Code.Relation.Nullary.Reflects

-- LangEq._≟ⁱ_
d__'8799''8305'__6 ::
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d__'8799''8305'__6 v0 v1
  = case coe v0 of
      MAlonzo.Code.Lang.C_ix_32 v2
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ix_32 v3
               -> let v4 = MAlonzo.Code.Ar.d__'8799''738'__70 (coe v2) (coe v3) in
                  coe
                    (case coe v4 of
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v5 v6
                         -> if coe v5
                              then coe
                                     seq (coe v6)
                                     (coe
                                        MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                                        (coe v5)
                                        (coe
                                           MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                                           erased))
                              else coe
                                     seq (coe v6)
                                     (coe
                                        MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                                        (coe v5)
                                        (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26))
                       _ -> MAlonzo.RTE.mazUnreachableError)
             MAlonzo.Code.Lang.C_ar_34 v3
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_ar_34 v2
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ix_32 v3
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_ar_34 v3
               -> let v4 = MAlonzo.Code.Ar.d__'8799''738'__70 (coe v2) (coe v3) in
                  coe
                    (case coe v4 of
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v5 v6
                         -> if coe v5
                              then coe
                                     seq (coe v6)
                                     (coe
                                        MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                                        (coe v5)
                                        (coe
                                           MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                                           erased))
                              else coe
                                     seq (coe v6)
                                     (coe
                                        MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                                        (coe v5)
                                        (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26))
                       _ -> MAlonzo.RTE.mazUnreachableError)
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq._≟ᵒ_
d__'8799''7506'__60 ::
  MAlonzo.Code.Lang.T_Bop_188 ->
  MAlonzo.Code.Lang.T_Bop_188 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d__'8799''7506'__60 v0 v1
  = case coe v0 of
      MAlonzo.Code.Lang.C_plus_190
        -> case coe v1 of
             MAlonzo.Code.Lang.C_plus_190
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
             MAlonzo.Code.Lang.C_mul_192
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_mul_192
        -> case coe v1 of
             MAlonzo.Code.Lang.C_plus_190
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_mul_192
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq._≟ᵘ_
d__'8799''7512'__66 ::
  MAlonzo.Code.Lang.T_Uop_194 ->
  MAlonzo.Code.Lang.T_Uop_194 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d__'8799''7512'__66 v0 v1
  = case coe v0 of
      MAlonzo.Code.Lang.C_logistic_196
        -> case coe v1 of
             MAlonzo.Code.Lang.C_logistic_196
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
             MAlonzo.Code.Lang.C_neg_198
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_exp_200
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_rectifier_202
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_squared_204
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_inverse_206
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_ind'45'positive_208
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_logarithm_210
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_neg_198
        -> case coe v1 of
             MAlonzo.Code.Lang.C_logistic_196
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_neg_198
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
             MAlonzo.Code.Lang.C_exp_200
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_rectifier_202
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_squared_204
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_inverse_206
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_ind'45'positive_208
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_logarithm_210
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_exp_200
        -> case coe v1 of
             MAlonzo.Code.Lang.C_logistic_196
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_neg_198
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_exp_200
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
             MAlonzo.Code.Lang.C_rectifier_202
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_squared_204
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_inverse_206
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_ind'45'positive_208
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_logarithm_210
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_rectifier_202
        -> case coe v1 of
             MAlonzo.Code.Lang.C_logistic_196
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_neg_198
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_exp_200
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_rectifier_202
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
             MAlonzo.Code.Lang.C_squared_204
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_inverse_206
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_ind'45'positive_208
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_logarithm_210
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_squared_204
        -> case coe v1 of
             MAlonzo.Code.Lang.C_logistic_196
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_neg_198
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_exp_200
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_rectifier_202
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_squared_204
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
             MAlonzo.Code.Lang.C_inverse_206
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_ind'45'positive_208
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_logarithm_210
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_inverse_206
        -> case coe v1 of
             MAlonzo.Code.Lang.C_logistic_196
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_neg_198
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_exp_200
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_rectifier_202
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_squared_204
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_inverse_206
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
             MAlonzo.Code.Lang.C_ind'45'positive_208
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_logarithm_210
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_ind'45'positive_208
        -> case coe v1 of
             MAlonzo.Code.Lang.C_logistic_196
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_neg_198
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_exp_200
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_rectifier_202
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_squared_204
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_inverse_206
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_ind'45'positive_208
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
             MAlonzo.Code.Lang.C_logarithm_210
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_logarithm_210
        -> case coe v1 of
             MAlonzo.Code.Lang.C_logistic_196
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_neg_198
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_exp_200
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_rectifier_202
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_squared_204
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_inverse_206
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_ind'45'positive_208
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
             MAlonzo.Code.Lang.C_logarithm_210
               -> coe
                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                    (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
                    (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isVar
d_isVar_72 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isVar_72 ~v0 ~v1 v2 = du_isVar_72 v2
du_isVar_72 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isVar_72 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3) erased))
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isZero
d_isZero_144 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isZero_144 ~v0 ~v1 v2 = du_isZero_144 v2
du_isZero_144 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isZero_144 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isOne
d_isOne_216 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isOne_216 ~v0 ~v1 v2 = du_isOne_216 v2
du_isOne_216 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isOne_216 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 erased)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isImap
d_isImap_296 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isImap_296 ~v0 ~v1 v2 = du_isImap_296 v2
du_isImap_296 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isImap_296 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v2)
                   (coe
                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3)
                      (coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 erased
                         (coe
                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) erased)))))
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isImaps
d_isImaps_404 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isImaps_404 ~v0 ~v1 v2 = du_isImaps_404 v2
du_isImaps_404 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isImaps_404 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3) erased))
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isZeroBut
d_isZeroBut_484 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isZeroBut_484 ~v0 ~v1 v2 = du_isZeroBut_484 v2
du_isZeroBut_484 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isZeroBut_484 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v2)
                   (coe
                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                      (coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v5)
                         (coe
                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v6) erased)))))
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isSels
d_isSels_564 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  [Integer] -> MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isSels_564 ~v0 ~v1 v2 v3 = du_isSels_564 v2 v3
du_isSels_564 ::
  MAlonzo.Code.Lang.T_E_214 ->
  [Integer] -> MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isSels_564 v0 v1
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v3 v4 v5
        -> let v6 = MAlonzo.Code.Ar.d__'8799''738'__70 (coe v3) (coe v1) in
           coe
             (case coe v6 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v7 v8
                  -> if coe v7
                       then coe
                              seq (coe v8)
                              (coe
                                 MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                                 (coe v7)
                                 (coe
                                    MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                                    (coe
                                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 erased
                                       (coe
                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                                          (coe
                                             MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v5)
                                             erased)))))
                       else coe
                              seq (coe v8)
                              (coe
                                 MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                                 (coe v7)
                                 (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26))
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_imap_226 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v3 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v2 v3 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v2 v4 v6 v7 v8
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v3 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v3 v4 v5 v7 v8 v9 v10
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v3 v4 v5 v7 v8 v9 v10
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v3 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v3 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq._.foo
d_foo_632 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  [Integer] ->
  MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14 ->
  MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20
d_foo_632 = erased
-- LangEq.isSel
d_isSel_792 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isSel_792 ~v0 ~v1 v2 = du_isSel_792 v2
du_isSel_792 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isSel_792 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v2)
                   (coe
                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                      (coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v5) erased))))
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isImapb
d_isImapb_906 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isImapb_906 ~v0 ~v1 v2 = du_isImapb_906 v2
du_isImapb_906 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isImapb_906 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v1)
                   (coe
                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v2)
                      (coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v5)
                         (coe
                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v6) erased)))))
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isSelb
d_isSelb_1022 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isSelb_1022 ~v0 ~v1 v2 = du_isSelb_1022 v2
du_isSelb_1022 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isSelb_1022 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v1)
                   (coe
                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3)
                      (coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v5)
                         (coe
                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v6)
                            (coe
                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v7) erased))))))
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isSum
d_isSum_1132 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isSum_1132 ~v0 ~v1 v2 = du_isSum_1132 v2
du_isSum_1132 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isSum_1132 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v2)
                   (coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) erased)))
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isMaximum
d_isMaximum_1208 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isMaximum_1208 ~v0 ~v1 v2 = du_isMaximum_1208 v2
du_isMaximum_1208 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isMaximum_1208 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v2)
                   (coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) erased)))
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isSlide
d_isSlide_1294 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isSlide_1294 ~v0 ~v1 v2 = du_isSlide_1294 v2
du_isSlide_1294 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isSlide_1294 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v2)
                   (coe
                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3)
                      (coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                         (coe
                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v6)
                            (coe
                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v7)
                               (coe
                                  MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v8)
                                  (coe
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v9)
                                     erased))))))))
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> let v5
                 = coe
                     MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                     (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                     (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26) in
           coe
             (case coe v3 of
                MAlonzo.Code.Lang.C_logistic_196
                  -> coe
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                       (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                       (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
                MAlonzo.Code.Lang.C_neg_198
                  -> coe
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                       (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                       (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
                _ -> coe v5)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isBackslide
d_isBackslide_1384 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isBackslide_1384 ~v0 ~v1 v2 = du_isBackslide_1384 v2
du_isBackslide_1384 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isBackslide_1384 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v2)
                   (coe
                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3)
                      (coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                         (coe
                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v6)
                            (coe
                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v7)
                               (coe
                                  MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v8)
                                  (coe
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v9)
                                     erased))))))))
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> let v5
                 = coe
                     MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                     (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                     (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26) in
           coe
             (case coe v3 of
                MAlonzo.Code.Lang.C_logistic_196
                  -> coe
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                       (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                       (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
                MAlonzo.Code.Lang.C_neg_198
                  -> coe
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                       (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                       (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
                _ -> coe v5)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isUn
d_isUn_1464 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isUn_1464 ~v0 ~v1 v2 = du_isUn_1464 v2
du_isUn_1464 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isUn_1464 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3)
                   (coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) erased)))
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isBin
d_isBin_1542 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isBin_1542 ~v0 ~v1 v2 = du_isBin_1542 v2
du_isBin_1542 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isBin_1542 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3)
                   (coe
                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                      (coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v5) erased))))
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> let v5
                 = coe
                     MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                     (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                     (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26) in
           coe
             (case coe v3 of
                MAlonzo.Code.Lang.C_logistic_196
                  -> coe
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                       (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                       (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
                MAlonzo.Code.Lang.C_neg_198
                  -> coe
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                       (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                       (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
                _ -> coe v5)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isScaledown
d_isScaledown_1622 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isScaledown_1622 ~v0 ~v1 v2 = du_isScaledown_1622 v2
du_isScaledown_1622 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isScaledown_1622 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3)
                   (coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) erased)))
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> let v5
                 = coe
                     MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                     (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                     (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26) in
           coe
             (case coe v3 of
                MAlonzo.Code.Lang.C_logistic_196
                  -> coe
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                       (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                       (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
                MAlonzo.Code.Lang.C_neg_198
                  -> coe
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                       (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                       (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
                _ -> coe v5)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.isLet
d_isLet_1704 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isLet_1704 ~v0 ~v1 v2 = du_isLet_1704 v2
du_isLet_1704 ::
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isLet_1704 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_var_216 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imaps_222 v3
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sels_224 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imap_226 v2 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sel_228 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_imapb_230 v1 v2 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_selb_232 v1 v3 v5 v6 v7
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_sum_234 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_zero'45'but_236 v2 v4 v5 v6
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_slide_238 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_backslide_240 v2 v3 v4 v6 v7 v8 v9
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_bin_242 v3 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_scaledown_244 v3 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      MAlonzo.Code.Lang.C_let'8242'_246 v2 v4 v5
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_true_10)
             (coe
                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22
                (coe
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v2)
                   (coe
                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                      (coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v5) erased))))
      MAlonzo.Code.Lang.C_un_248 v3 v4
        -> let v5
                 = coe
                     MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                     (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                     (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26) in
           coe
             (case coe v3 of
                MAlonzo.Code.Lang.C_logistic_196
                  -> coe
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                       (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                       (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
                MAlonzo.Code.Lang.C_neg_198
                  -> coe
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
                       (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
                       (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
                _ -> coe v5)
      MAlonzo.Code.Lang.C_maximum_250 v2 v4
        -> coe
             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32
             (coe MAlonzo.Code.Agda.Builtin.Bool.C_false_8)
             (coe MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.unvar
d_unvar_1782 ::
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_unvar_1782 = erased
-- LangEq.*≈-uniq
d_'42''8776''45'uniq_1788 ::
  [Integer] ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Ar.T_Pointw'8323'_990 ->
  MAlonzo.Code.Ar.T_Pointw'8323'_990 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''8776''45'uniq_1788 = erased
-- LangEq.+≈-uniq
d_'43''8776''45'uniq_1814 ::
  [Integer] ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Ar.T_Pointw'8323'_990 ->
  MAlonzo.Code.Ar.T_Pointw'8323'_990 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'43''8776''45'uniq_1814 = erased
-- LangEq.suc≈-uniq
d_suc'8776''45'uniq_1840 ::
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Ar.T_Pointw'8322'_968 ->
  MAlonzo.Code.Ar.T_Pointw'8322'_968 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_suc'8776''45'uniq_1840 = erased
-- LangEq._≟ᵉ_
d__'8799''7497'__1854 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  Maybe MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d__'8799''7497'__1854 v0 v1 v2 v3
  = case coe v2 of
      MAlonzo.Code.Lang.C_var_216 v6
        -> let v7 = coe du_isVar_72 (coe v3) in
           coe
             (case coe v7 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v8 v9
                  -> if coe v8
                       then case coe v9 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v10
                                -> case coe v10 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v11 v12
                                       -> let v13
                                                = coe
                                                    MAlonzo.Code.Lang.du_eq'63'_142 (coe v0)
                                                    (coe v6) (coe v11) in
                                          coe
                                            (case coe v13 of
                                               MAlonzo.Code.Lang.C_veq_130
                                                 -> coe
                                                      MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                      erased
                                               MAlonzo.Code.Lang.C_neq_136 v18
                                                 -> coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                               _ -> MAlonzo.RTE.mazUnreachableError)
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v9) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_zero_218
        -> let v6 = coe du_isZero_144 (coe v3) in
           coe
             (case coe v6 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v7 v8
                  -> if coe v7
                       then coe
                              seq (coe v8) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_just_16 erased)
                       else coe
                              seq (coe v8) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_one_220
        -> let v6 = coe du_isOne_216 (coe v3) in
           coe
             (case coe v6 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v7 v8
                  -> if coe v7
                       then coe
                              seq (coe v8) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_just_16 erased)
                       else coe
                              seq (coe v8) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_imaps_222 v6
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v7
               -> let v8 = coe du_isImaps_404 (coe v3) in
                  coe
                    (case coe v8 of
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v9 v10
                         -> if coe v9
                              then case coe v10 of
                                     MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v11
                                       -> case coe v11 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                              -> coe
                                                   MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                   (coe
                                                      d__'8799''7497'__1854
                                                      (coe
                                                         MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                                         (coe MAlonzo.Code.Lang.C_ix_32 (coe v7)))
                                                      (coe
                                                         MAlonzo.Code.Lang.C_ar_34
                                                         (coe
                                                            MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                                                      (coe v6) (coe v12))
                                                   (coe
                                                      (\ v14 ->
                                                         coe
                                                           MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                           erased))
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              else coe
                                     seq (coe v10)
                                     (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                       _ -> MAlonzo.RTE.mazUnreachableError)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_sels_224 v5 v6 v7
        -> let v8 = coe du_isSels_564 (coe v3) (coe v5) in
           coe
             (case coe v8 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v9 v10
                  -> if coe v9
                       then case coe v10 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v11
                                -> case coe v11 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                       -> case coe v13 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                              -> case coe v15 of
                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                                                     -> coe
                                                          MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                          (coe
                                                             d__'8799''7497'__1854 (coe v0)
                                                             (coe
                                                                MAlonzo.Code.Lang.C_ar_34 (coe v5))
                                                             (coe v6) (coe v14))
                                                          (coe
                                                             (\ v18 ->
                                                                coe
                                                                  MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                  (coe
                                                                     d__'8799''7497'__1854 (coe v0)
                                                                     (coe
                                                                        MAlonzo.Code.Lang.C_ix_32
                                                                        (coe v5))
                                                                     (coe v7) (coe v16))
                                                                  (coe
                                                                     (\ v19 ->
                                                                        coe
                                                                          MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                          erased))))
                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v10) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_imap_226 v5 v6 v7
        -> let v8 = coe du_isImap_296 (coe v3) in
           coe
             (case coe v8 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v9 v10
                  -> if coe v9
                       then case coe v10 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v11
                                -> case coe v11 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                       -> case coe v13 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                              -> case coe v15 of
                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                                                     -> case coe v17 of
                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v18 v19
                                                            -> let v20
                                                                     = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                         (coe v5) (coe v12) in
                                                               coe
                                                                 (case coe v20 of
                                                                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v21 v22
                                                                      -> if coe v21
                                                                           then coe
                                                                                  seq (coe v22)
                                                                                  (coe
                                                                                     MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                     (coe
                                                                                        d__'8799''7497'__1854
                                                                                        (coe
                                                                                           MAlonzo.Code.Lang.C__'9657'__40
                                                                                           (coe v0)
                                                                                           (coe
                                                                                              MAlonzo.Code.Lang.C_ix_32
                                                                                              (coe
                                                                                                 v12)))
                                                                                        (coe
                                                                                           MAlonzo.Code.Lang.C_ar_34
                                                                                           (coe
                                                                                              v14))
                                                                                        (coe v7)
                                                                                        (coe v18))
                                                                                     (coe
                                                                                        (\ v23 ->
                                                                                           coe
                                                                                             MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                                             erased)))
                                                                           else coe
                                                                                  seq (coe v22)
                                                                                  (coe
                                                                                     MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                                    _ -> MAlonzo.RTE.mazUnreachableError)
                                                          _ -> MAlonzo.RTE.mazUnreachableError
                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v10) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_sel_228 v5 v7 v8
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v9
               -> let v10 = coe du_isSel_792 (coe v3) in
                  coe
                    (case coe v10 of
                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v11 v12
                         -> if coe v11
                              then case coe v12 of
                                     MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v13
                                       -> case coe v13 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                              -> case coe v15 of
                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                                                     -> case coe v17 of
                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v18 v19
                                                            -> let v20
                                                                     = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                         (coe v5) (coe v14) in
                                                               coe
                                                                 (case coe v20 of
                                                                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v21 v22
                                                                      -> if coe v21
                                                                           then coe
                                                                                  seq (coe v22)
                                                                                  (coe
                                                                                     MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                     (coe
                                                                                        d__'8799''7497'__1854
                                                                                        (coe v0)
                                                                                        (coe
                                                                                           MAlonzo.Code.Lang.C_ar_34
                                                                                           (coe
                                                                                              MAlonzo.Code.Data.List.Base.du__'43''43'__32
                                                                                              (coe
                                                                                                 v14)
                                                                                              (coe
                                                                                                 v9)))
                                                                                        (coe v7)
                                                                                        (coe v16))
                                                                                     (coe
                                                                                        (\ v23 ->
                                                                                           coe
                                                                                             MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                             (coe
                                                                                                d__'8799''7497'__1854
                                                                                                (coe
                                                                                                   v0)
                                                                                                (coe
                                                                                                   MAlonzo.Code.Lang.C_ix_32
                                                                                                   (coe
                                                                                                      v14))
                                                                                                (coe
                                                                                                   v8)
                                                                                                (coe
                                                                                                   v18))
                                                                                             (coe
                                                                                                (\ v24 ->
                                                                                                   coe
                                                                                                     MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                                                     erased)))))
                                                                           else coe
                                                                                  seq (coe v22)
                                                                                  (coe
                                                                                     MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                                    _ -> MAlonzo.RTE.mazUnreachableError)
                                                          _ -> MAlonzo.RTE.mazUnreachableError
                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              else coe
                                     seq (coe v12)
                                     (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                       _ -> MAlonzo.RTE.mazUnreachableError)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_imapb_230 v4 v5 v8 v9
        -> let v10 = coe du_isImapb_906 (coe v3) in
           coe
             (case coe v10 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v11 v12
                  -> if coe v11
                       then case coe v12 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v13
                                -> case coe v13 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                       -> case coe v15 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                                              -> case coe v17 of
                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v18 v19
                                                     -> case coe v19 of
                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v20 v21
                                                            -> let v22
                                                                     = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                         (coe v4) (coe v14) in
                                                               coe
                                                                 (case coe v22 of
                                                                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v23 v24
                                                                      -> if coe v23
                                                                           then coe
                                                                                  seq (coe v24)
                                                                                  (let v25
                                                                                         = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                                             (coe
                                                                                                v5)
                                                                                             (coe
                                                                                                v16) in
                                                                                   coe
                                                                                     (case coe
                                                                                             v25 of
                                                                                        MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v26 v27
                                                                                          -> if coe
                                                                                                  v26
                                                                                               then coe
                                                                                                      seq
                                                                                                      (coe
                                                                                                         v27)
                                                                                                      (coe
                                                                                                         MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                                         (coe
                                                                                                            d__'8799''7497'__1854
                                                                                                            (coe
                                                                                                               MAlonzo.Code.Lang.C__'9657'__40
                                                                                                               (coe
                                                                                                                  v0)
                                                                                                               (coe
                                                                                                                  MAlonzo.Code.Lang.C_ix_32
                                                                                                                  (coe
                                                                                                                     v14)))
                                                                                                            (coe
                                                                                                               MAlonzo.Code.Lang.C_ar_34
                                                                                                               (coe
                                                                                                                  v16))
                                                                                                            (coe
                                                                                                               v9)
                                                                                                            (coe
                                                                                                               v20))
                                                                                                         (coe
                                                                                                            (\ v28 ->
                                                                                                               coe
                                                                                                                 MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                                                                 erased)))
                                                                                               else coe
                                                                                                      seq
                                                                                                      (coe
                                                                                                         v27)
                                                                                                      (coe
                                                                                                         MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                                                        _ -> MAlonzo.RTE.mazUnreachableError))
                                                                           else coe
                                                                                  seq (coe v24)
                                                                                  (coe
                                                                                     MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                                    _ -> MAlonzo.RTE.mazUnreachableError)
                                                          _ -> MAlonzo.RTE.mazUnreachableError
                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v12) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_selb_232 v4 v6 v8 v9 v10
        -> let v11 = coe du_isSelb_1022 (coe v3) in
           coe
             (case coe v11 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v12 v13
                  -> if coe v12
                       then case coe v13 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v14
                                -> case coe v14 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v15 v16
                                       -> case coe v16 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v17 v18
                                              -> case coe v18 of
                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v19 v20
                                                     -> case coe v20 of
                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v21 v22
                                                            -> case coe v22 of
                                                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v23 v24
                                                                   -> let v25
                                                                            = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                                (coe v4)
                                                                                (coe v15) in
                                                                      coe
                                                                        (case coe v25 of
                                                                           MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v26 v27
                                                                             -> if coe v26
                                                                                  then coe
                                                                                         seq
                                                                                         (coe v27)
                                                                                         (let v28
                                                                                                = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                                                    (coe
                                                                                                       v6)
                                                                                                    (coe
                                                                                                       v17) in
                                                                                          coe
                                                                                            (case coe
                                                                                                    v28 of
                                                                                               MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v29 v30
                                                                                                 -> if coe
                                                                                                         v29
                                                                                                      then coe
                                                                                                             seq
                                                                                                             (coe
                                                                                                                v30)
                                                                                                             (coe
                                                                                                                MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                                                (coe
                                                                                                                   d__'8799''7497'__1854
                                                                                                                   (coe
                                                                                                                      v0)
                                                                                                                   (coe
                                                                                                                      MAlonzo.Code.Lang.C_ar_34
                                                                                                                      (coe
                                                                                                                         v17))
                                                                                                                   (coe
                                                                                                                      v9)
                                                                                                                   (coe
                                                                                                                      v21))
                                                                                                                (coe
                                                                                                                   (\ v31 ->
                                                                                                                      coe
                                                                                                                        MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                                                        (coe
                                                                                                                           d__'8799''7497'__1854
                                                                                                                           (coe
                                                                                                                              v0)
                                                                                                                           (coe
                                                                                                                              MAlonzo.Code.Lang.C_ix_32
                                                                                                                              (coe
                                                                                                                                 v15))
                                                                                                                           (coe
                                                                                                                              v10)
                                                                                                                           (coe
                                                                                                                              v23))
                                                                                                                        (coe
                                                                                                                           (\ v32 ->
                                                                                                                              coe
                                                                                                                                MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                                                                                erased)))))
                                                                                                      else coe
                                                                                                             seq
                                                                                                             (coe
                                                                                                                v30)
                                                                                                             (coe
                                                                                                                MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                                                               _ -> MAlonzo.RTE.mazUnreachableError))
                                                                                  else coe
                                                                                         seq
                                                                                         (coe v27)
                                                                                         (coe
                                                                                            MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                                           _ -> MAlonzo.RTE.mazUnreachableError)
                                                                 _ -> MAlonzo.RTE.mazUnreachableError
                                                          _ -> MAlonzo.RTE.mazUnreachableError
                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v13) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_sum_234 v5 v7
        -> let v8 = coe du_isSum_1132 (coe v3) in
           coe
             (case coe v8 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v9 v10
                  -> if coe v9
                       then case coe v10 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v11
                                -> case coe v11 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                       -> case coe v13 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                              -> let v16
                                                       = MAlonzo.Code.Ar.d__'8799''738'__70
                                                           (coe v5) (coe v12) in
                                                 coe
                                                   (case coe v16 of
                                                      MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v17 v18
                                                        -> if coe v17
                                                             then coe
                                                                    seq (coe v18)
                                                                    (coe
                                                                       MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                       (coe
                                                                          d__'8799''7497'__1854
                                                                          (coe
                                                                             MAlonzo.Code.Lang.C__'9657'__40
                                                                             (coe v0)
                                                                             (coe
                                                                                MAlonzo.Code.Lang.C_ix_32
                                                                                (coe v12)))
                                                                          (coe v1) (coe v7)
                                                                          (coe v14))
                                                                       (coe
                                                                          (\ v19 ->
                                                                             coe
                                                                               MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                               erased)))
                                                             else coe
                                                                    seq (coe v18)
                                                                    (coe
                                                                       MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                      _ -> MAlonzo.RTE.mazUnreachableError)
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v10) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_zero'45'but_236 v5 v7 v8 v9
        -> let v10 = coe du_isZeroBut_484 (coe v3) in
           coe
             (case coe v10 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v11 v12
                  -> if coe v11
                       then case coe v12 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v13
                                -> case coe v13 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                       -> case coe v15 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                                              -> case coe v17 of
                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v18 v19
                                                     -> case coe v19 of
                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v20 v21
                                                            -> let v22
                                                                     = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                         (coe v5) (coe v14) in
                                                               coe
                                                                 (case coe v22 of
                                                                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v23 v24
                                                                      -> if coe v23
                                                                           then coe
                                                                                  seq (coe v24)
                                                                                  (coe
                                                                                     MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                     (coe
                                                                                        d__'8799''7497'__1854
                                                                                        (coe v0)
                                                                                        (coe
                                                                                           MAlonzo.Code.Lang.C_ix_32
                                                                                           (coe
                                                                                              v14))
                                                                                        (coe v7)
                                                                                        (coe v16))
                                                                                     (coe
                                                                                        (\ v25 ->
                                                                                           coe
                                                                                             MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                             (coe
                                                                                                d__'8799''7497'__1854
                                                                                                (coe
                                                                                                   v0)
                                                                                                (coe
                                                                                                   MAlonzo.Code.Lang.C_ix_32
                                                                                                   (coe
                                                                                                      v14))
                                                                                                (coe
                                                                                                   v8)
                                                                                                (coe
                                                                                                   v18))
                                                                                             (coe
                                                                                                (\ v26 ->
                                                                                                   coe
                                                                                                     MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                                     (coe
                                                                                                        d__'8799''7497'__1854
                                                                                                        (coe
                                                                                                           v0)
                                                                                                        (coe
                                                                                                           v1)
                                                                                                        (coe
                                                                                                           v9)
                                                                                                        (coe
                                                                                                           v20))
                                                                                                     (coe
                                                                                                        (\ v27 ->
                                                                                                           coe
                                                                                                             MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                                                             erased)))))))
                                                                           else coe
                                                                                  seq (coe v24)
                                                                                  (coe
                                                                                     MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                                    _ -> MAlonzo.RTE.mazUnreachableError)
                                                          _ -> MAlonzo.RTE.mazUnreachableError
                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v12) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_slide_238 v5 v6 v7 v9 v10 v11 v12
        -> let v13 = coe du_isSlide_1294 (coe v3) in
           coe
             (case coe v13 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v14 v15
                  -> if coe v14
                       then case coe v15 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v16
                                -> case coe v16 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v17 v18
                                       -> case coe v18 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v19 v20
                                              -> case coe v20 of
                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v21 v22
                                                     -> case coe v22 of
                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v23 v24
                                                            -> case coe v24 of
                                                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v25 v26
                                                                   -> case coe v26 of
                                                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v27 v28
                                                                          -> case coe v28 of
                                                                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v29 v30
                                                                                 -> let v31
                                                                                          = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                                              (coe
                                                                                                 v5)
                                                                                              (coe
                                                                                                 v17) in
                                                                                    coe
                                                                                      (let v32
                                                                                             = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                                                 (coe
                                                                                                    v6)
                                                                                                 (coe
                                                                                                    v19) in
                                                                                       coe
                                                                                         (let v33
                                                                                                = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                                                    (coe
                                                                                                       v7)
                                                                                                    (coe
                                                                                                       v21) in
                                                                                          coe
                                                                                            (case coe
                                                                                                    v31 of
                                                                                               MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v34 v35
                                                                                                 -> case coe
                                                                                                           v34 of
                                                                                                      MAlonzo.Code.Agda.Builtin.Bool.C_true_10
                                                                                                        -> case coe
                                                                                                                  v35 of
                                                                                                             MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v36
                                                                                                               -> case coe
                                                                                                                         v32 of
                                                                                                                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v37 v38
                                                                                                                      -> case coe
                                                                                                                                v37 of
                                                                                                                           MAlonzo.Code.Agda.Builtin.Bool.C_true_10
                                                                                                                             -> case coe
                                                                                                                                       v38 of
                                                                                                                                  MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v39
                                                                                                                                    -> case coe
                                                                                                                                              v33 of
                                                                                                                                         MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v40 v41
                                                                                                                                           -> case coe
                                                                                                                                                     v40 of
                                                                                                                                                MAlonzo.Code.Agda.Builtin.Bool.C_true_10
                                                                                                                                                  -> case coe
                                                                                                                                                            v41 of
                                                                                                                                                       MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v42
                                                                                                                                                         -> coe
                                                                                                                                                              MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                                                                                              (coe
                                                                                                                                                                 d__'8799''7497'__1854
                                                                                                                                                                 (coe
                                                                                                                                                                    v0)
                                                                                                                                                                 (coe
                                                                                                                                                                    MAlonzo.Code.Lang.C_ix_32
                                                                                                                                                                    (coe
                                                                                                                                                                       v17))
                                                                                                                                                                 (coe
                                                                                                                                                                    v9)
                                                                                                                                                                 (coe
                                                                                                                                                                    v23))
                                                                                                                                                              (coe
                                                                                                                                                                 (\ v43 ->
                                                                                                                                                                    coe
                                                                                                                                                                      MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                                                                                                      (coe
                                                                                                                                                                         d__'8799''7497'__1854
                                                                                                                                                                         (coe
                                                                                                                                                                            v0)
                                                                                                                                                                         (coe
                                                                                                                                                                            MAlonzo.Code.Lang.C_ar_34
                                                                                                                                                                            (coe
                                                                                                                                                                               v21))
                                                                                                                                                                         (coe
                                                                                                                                                                            v11)
                                                                                                                                                                         (coe
                                                                                                                                                                            v27))
                                                                                                                                                                      (coe
                                                                                                                                                                         (\ v44 ->
                                                                                                                                                                            coe
                                                                                                                                                                              MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                                                                                                                              erased))))
                                                                                                                                                       _ -> coe
                                                                                                                                                              MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                                                                                _ -> coe
                                                                                                                                                       MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                                                                         _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                  _ -> coe
                                                                                                                                         MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                                                           _ -> coe
                                                                                                                                  MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                                                    _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                             _ -> coe
                                                                                                                    MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                                      _ -> coe
                                                                                                             MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                               _ -> MAlonzo.RTE.mazUnreachableError)))
                                                                               _ -> MAlonzo.RTE.mazUnreachableError
                                                                        _ -> MAlonzo.RTE.mazUnreachableError
                                                                 _ -> MAlonzo.RTE.mazUnreachableError
                                                          _ -> MAlonzo.RTE.mazUnreachableError
                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v15) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_backslide_240 v5 v6 v7 v9 v10 v11 v12
        -> let v13 = coe du_isBackslide_1384 (coe v3) in
           coe
             (case coe v13 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v14 v15
                  -> if coe v14
                       then case coe v15 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v16
                                -> case coe v16 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v17 v18
                                       -> case coe v18 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v19 v20
                                              -> case coe v20 of
                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v21 v22
                                                     -> case coe v22 of
                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v23 v24
                                                            -> case coe v24 of
                                                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v25 v26
                                                                   -> case coe v26 of
                                                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v27 v28
                                                                          -> case coe v28 of
                                                                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v29 v30
                                                                                 -> let v31
                                                                                          = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                                              (coe
                                                                                                 v5)
                                                                                              (coe
                                                                                                 v17) in
                                                                                    coe
                                                                                      (let v32
                                                                                             = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                                                 (coe
                                                                                                    v6)
                                                                                                 (coe
                                                                                                    v19) in
                                                                                       coe
                                                                                         (let v33
                                                                                                = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                                                    (coe
                                                                                                       v7)
                                                                                                    (coe
                                                                                                       v21) in
                                                                                          coe
                                                                                            (case coe
                                                                                                    v31 of
                                                                                               MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v34 v35
                                                                                                 -> case coe
                                                                                                           v34 of
                                                                                                      MAlonzo.Code.Agda.Builtin.Bool.C_true_10
                                                                                                        -> case coe
                                                                                                                  v35 of
                                                                                                             MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v36
                                                                                                               -> case coe
                                                                                                                         v32 of
                                                                                                                    MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v37 v38
                                                                                                                      -> case coe
                                                                                                                                v37 of
                                                                                                                           MAlonzo.Code.Agda.Builtin.Bool.C_true_10
                                                                                                                             -> case coe
                                                                                                                                       v38 of
                                                                                                                                  MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v39
                                                                                                                                    -> case coe
                                                                                                                                              v33 of
                                                                                                                                         MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v40 v41
                                                                                                                                           -> case coe
                                                                                                                                                     v40 of
                                                                                                                                                MAlonzo.Code.Agda.Builtin.Bool.C_true_10
                                                                                                                                                  -> case coe
                                                                                                                                                            v41 of
                                                                                                                                                       MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v42
                                                                                                                                                         -> coe
                                                                                                                                                              MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                                                                                              (coe
                                                                                                                                                                 d__'8799''7497'__1854
                                                                                                                                                                 (coe
                                                                                                                                                                    v0)
                                                                                                                                                                 (coe
                                                                                                                                                                    MAlonzo.Code.Lang.C_ix_32
                                                                                                                                                                    (coe
                                                                                                                                                                       v17))
                                                                                                                                                                 (coe
                                                                                                                                                                    v9)
                                                                                                                                                                 (coe
                                                                                                                                                                    v23))
                                                                                                                                                              (coe
                                                                                                                                                                 (\ v43 ->
                                                                                                                                                                    coe
                                                                                                                                                                      MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                                                                                                      (coe
                                                                                                                                                                         d__'8799''7497'__1854
                                                                                                                                                                         (coe
                                                                                                                                                                            v0)
                                                                                                                                                                         (coe
                                                                                                                                                                            MAlonzo.Code.Lang.C_ar_34
                                                                                                                                                                            (coe
                                                                                                                                                                               v19))
                                                                                                                                                                         (coe
                                                                                                                                                                            v10)
                                                                                                                                                                         (coe
                                                                                                                                                                            v25))
                                                                                                                                                                      (coe
                                                                                                                                                                         (\ v44 ->
                                                                                                                                                                            coe
                                                                                                                                                                              MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                                                                                                                              erased))))
                                                                                                                                                       _ -> coe
                                                                                                                                                              MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                                                                                _ -> coe
                                                                                                                                                       MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                                                                         _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                  _ -> coe
                                                                                                                                         MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                                                           _ -> coe
                                                                                                                                  MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                                                    _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                             _ -> coe
                                                                                                                    MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                                      _ -> coe
                                                                                                             MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                                               _ -> MAlonzo.RTE.mazUnreachableError)))
                                                                               _ -> MAlonzo.RTE.mazUnreachableError
                                                                        _ -> MAlonzo.RTE.mazUnreachableError
                                                                 _ -> MAlonzo.RTE.mazUnreachableError
                                                          _ -> MAlonzo.RTE.mazUnreachableError
                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v15) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_bin_242 v6 v7 v8
        -> let v9 = coe du_isBin_1542 (coe v3) in
           coe
             (case coe v9 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v10 v11
                  -> if coe v10
                       then case coe v11 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v12
                                -> case coe v12 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v13 v14
                                       -> case coe v14 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v15 v16
                                              -> case coe v16 of
                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v17 v18
                                                     -> let v19
                                                              = d__'8799''7506'__60
                                                                  (coe v6) (coe v13) in
                                                        coe
                                                          (case coe v19 of
                                                             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v20 v21
                                                               -> if coe v20
                                                                    then coe
                                                                           seq (coe v21)
                                                                           (coe
                                                                              MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                              (coe
                                                                                 d__'8799''7497'__1854
                                                                                 (coe v0) (coe v1)
                                                                                 (coe v7) (coe v15))
                                                                              (coe
                                                                                 (\ v22 ->
                                                                                    coe
                                                                                      MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                      (coe
                                                                                         d__'8799''7497'__1854
                                                                                         (coe v0)
                                                                                         (coe v1)
                                                                                         (coe v8)
                                                                                         (coe v17))
                                                                                      (coe
                                                                                         (\ v23 ->
                                                                                            coe
                                                                                              MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                                              erased)))))
                                                                    else coe
                                                                           seq (coe v21)
                                                                           (coe
                                                                              MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                             _ -> MAlonzo.RTE.mazUnreachableError)
                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v11) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_scaledown_244 v6 v7
        -> let v8 = coe du_isScaledown_1622 (coe v3) in
           coe
             (case coe v8 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v9 v10
                  -> if coe v9
                       then case coe v10 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v11
                                -> case coe v11 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                       -> case coe v13 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                              -> let v16
                                                       = coe
                                                           MAlonzo.Code.Relation.Nullary.Decidable.Core.du_map'8242'_178
                                                           erased
                                                           (\ v16 ->
                                                              coe
                                                                MAlonzo.Code.Data.Nat.Properties.du_'8801''8658''8801''7495'_2786
                                                                (coe v6))
                                                           (coe
                                                              MAlonzo.Code.Relation.Nullary.Decidable.Core.d_T'63'_72
                                                              (coe eqInt (coe v6) (coe v12))) in
                                                 coe
                                                   (case coe v16 of
                                                      MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v17 v18
                                                        -> if coe v17
                                                             then coe
                                                                    seq (coe v18)
                                                                    (coe
                                                                       MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                       (coe
                                                                          d__'8799''7497'__1854
                                                                          (coe v0) (coe v1) (coe v7)
                                                                          (coe v14))
                                                                       (coe
                                                                          (\ v19 ->
                                                                             coe
                                                                               MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                               erased)))
                                                             else coe
                                                                    seq (coe v18)
                                                                    (coe
                                                                       MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                      _ -> MAlonzo.RTE.mazUnreachableError)
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v10) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_let'8242'_246 v5 v7 v8
        -> let v9 = coe du_isLet_1704 (coe v3) in
           coe
             (case coe v9 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v10 v11
                  -> if coe v10
                       then case coe v11 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v12
                                -> case coe v12 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v13 v14
                                       -> case coe v14 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v15 v16
                                              -> case coe v16 of
                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v17 v18
                                                     -> let v19
                                                              = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                  (coe v5) (coe v13) in
                                                        coe
                                                          (case coe v19 of
                                                             MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v20 v21
                                                               -> if coe v20
                                                                    then coe
                                                                           seq (coe v21)
                                                                           (coe
                                                                              MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                              (coe
                                                                                 d__'8799''7497'__1854
                                                                                 (coe v0)
                                                                                 (coe
                                                                                    MAlonzo.Code.Lang.C_ar_34
                                                                                    (coe v13))
                                                                                 (coe v7) (coe v15))
                                                                              (coe
                                                                                 (\ v22 ->
                                                                                    coe
                                                                                      MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                                      (coe
                                                                                         d__'8799''7497'__1854
                                                                                         (coe
                                                                                            MAlonzo.Code.Lang.C__'9657'__40
                                                                                            (coe v0)
                                                                                            (coe
                                                                                               MAlonzo.Code.Lang.C_ar_34
                                                                                               (coe
                                                                                                  v13)))
                                                                                         (coe v1)
                                                                                         (coe v8)
                                                                                         (coe v17))
                                                                                      (coe
                                                                                         (\ v23 ->
                                                                                            coe
                                                                                              MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                                              erased)))))
                                                                    else coe
                                                                           seq (coe v21)
                                                                           (coe
                                                                              MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                             _ -> MAlonzo.RTE.mazUnreachableError)
                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v11) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_un_248 v6 v7
        -> let v8 = coe du_isUn_1464 (coe v3) in
           coe
             (case coe v8 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v9 v10
                  -> if coe v9
                       then case coe v10 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v11
                                -> case coe v11 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                       -> case coe v13 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                              -> let v16 = d__'8799''7512'__66 (coe v6) (coe v12) in
                                                 coe
                                                   (case coe v16 of
                                                      MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v17 v18
                                                        -> if coe v17
                                                             then coe
                                                                    seq (coe v18)
                                                                    (coe
                                                                       d__'8799''7497'__1854
                                                                       (coe v0) (coe v1) (coe v7)
                                                                       (coe v14))
                                                             else coe
                                                                    seq (coe v18)
                                                                    (coe
                                                                       MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                      _ -> MAlonzo.RTE.mazUnreachableError)
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v10) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_maximum_250 v5 v7
        -> let v8 = coe du_isMaximum_1208 (coe v3) in
           coe
             (case coe v8 of
                MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v9 v10
                  -> if coe v9
                       then case coe v10 of
                              MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v11
                                -> case coe v11 of
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                       -> case coe v13 of
                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                              -> let v16
                                                       = MAlonzo.Code.Ar.d__'8799''738'__70
                                                           (coe v5) (coe v12) in
                                                 coe
                                                   (case coe v16 of
                                                      MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v17 v18
                                                        -> if coe v17
                                                             then coe
                                                                    seq (coe v18)
                                                                    (coe
                                                                       MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                                                       (coe
                                                                          d__'8799''7497'__1854
                                                                          (coe
                                                                             MAlonzo.Code.Lang.C__'9657'__40
                                                                             (coe v0)
                                                                             (coe
                                                                                MAlonzo.Code.Lang.C_ix_32
                                                                                (coe v12)))
                                                                          (coe v1) (coe v7)
                                                                          (coe v14))
                                                                       (coe
                                                                          (\ v19 ->
                                                                             coe
                                                                               MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                                                               erased)))
                                                             else coe
                                                                    seq (coe v18)
                                                                    (coe
                                                                       MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                                                      _ -> MAlonzo.RTE.mazUnreachableError)
                                            _ -> MAlonzo.RTE.mazUnreachableError
                                     _ -> MAlonzo.RTE.mazUnreachableError
                              _ -> MAlonzo.RTE.mazUnreachableError
                       else coe
                              seq (coe v10) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
                _ -> MAlonzo.RTE.mazUnreachableError)
      _ -> MAlonzo.RTE.mazUnreachableError
-- LangEq.e-eq?
d_e'45'eq'63'_3218 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  Maybe MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14
d_e'45'eq'63'_3218 v0 v1 v2 v3 v4
  = let v5 = d__'8799''8305'__6 (coe v1) (coe v2) in
    coe
      (case coe v5 of
         MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v6 v7
           -> if coe v6
                then coe
                       seq (coe v7)
                       (coe
                          MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                          (coe d__'8799''7497'__1854 (coe v0) (coe v1) (coe v3) (coe v4))
                          (coe
                             (\ v8 ->
                                coe
                                  MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                  (coe
                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 erased (coe v8)))))
                else coe
                       seq (coe v7) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18)
         _ -> MAlonzo.RTE.mazUnreachableError)
-- LangEq.IsE₂
d_IsE'8322'_3264 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 -> MAlonzo.Code.Lang.T_E_214 -> ()
d_IsE'8322'_3264 = erased
-- LangEq.isE₂
d_isE'8322'_3600 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d_isE'8322'_3600 ~v0 ~v1 v2 v3 v4 = du_isE'8322'_3600 v2 v3 v4
du_isE'8322'_3600 ::
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
du_isE'8322'_3600 v0 v1 v2
  = case coe v0 of
      MAlonzo.Code.Lang.C_ix_32 v3
        -> coe seq (coe v1) (coe du_isVar_72 (coe v2))
      MAlonzo.Code.Lang.C_ar_34 v3
        -> case coe v1 of
             MAlonzo.Code.Lang.C_var_216 v6 -> coe du_isVar_72 (coe v2)
             MAlonzo.Code.Lang.C_zero_218 -> coe du_isZero_144 (coe v2)
             MAlonzo.Code.Lang.C_one_220 -> coe du_isOne_216 (coe v2)
             MAlonzo.Code.Lang.C_imaps_222 v6 -> coe du_isImaps_404 (coe v2)
             MAlonzo.Code.Lang.C_sels_224 v5 v6 v7
               -> coe du_isSels_564 (coe v2) (coe MAlonzo.Code.Lang.d_unit_212)
             MAlonzo.Code.Lang.C_imap_226 v5 v6 v7 -> coe du_isImap_296 (coe v2)
             MAlonzo.Code.Lang.C_sel_228 v5 v7 v8 -> coe du_isSel_792 (coe v2)
             MAlonzo.Code.Lang.C_imapb_230 v4 v5 v8 v9
               -> coe du_isImapb_906 (coe v2)
             MAlonzo.Code.Lang.C_selb_232 v4 v6 v8 v9 v10
               -> coe du_isSelb_1022 (coe v2)
             MAlonzo.Code.Lang.C_sum_234 v5 v7 -> coe du_isSum_1132 (coe v2)
             MAlonzo.Code.Lang.C_zero'45'but_236 v5 v7 v8 v9
               -> coe du_isZeroBut_484 (coe v2)
             MAlonzo.Code.Lang.C_slide_238 v5 v6 v7 v9 v10 v11 v12
               -> coe du_isSlide_1294 (coe v2)
             MAlonzo.Code.Lang.C_backslide_240 v5 v6 v7 v9 v10 v11 v12
               -> coe du_isBackslide_1384 (coe v2)
             MAlonzo.Code.Lang.C_bin_242 v6 v7 v8 -> coe du_isBin_1542 (coe v2)
             MAlonzo.Code.Lang.C_scaledown_244 v6 v7
               -> coe du_isScaledown_1622 (coe v2)
             MAlonzo.Code.Lang.C_let'8242'_246 v5 v7 v8
               -> coe du_isLet_1704 (coe v2)
             MAlonzo.Code.Lang.C_un_248 v6 v7 -> coe du_isUn_1464 (coe v2)
             MAlonzo.Code.Lang.C_maximum_250 v5 v7
               -> coe du_isMaximum_1208 (coe v2)
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
