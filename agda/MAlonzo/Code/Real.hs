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

module MAlonzo.Code.Real where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.Equality
import qualified MAlonzo.Code.Data.Irrelevant
import qualified MAlonzo.Code.Relation.Nullary.Decidable.Core

-- Real.Real
d_Real_2 = ()
data T_Real_2
  = C_constructor_64 (Integer -> AgdaAny) AgdaAny
                     (AgdaAny -> AgdaAny -> AgdaAny) (AgdaAny -> AgdaAny -> AgdaAny)
                     (AgdaAny -> AgdaAny -> AgdaAny) (AgdaAny -> AgdaAny -> AgdaAny)
                     (AgdaAny -> AgdaAny) (AgdaAny -> AgdaAny) (AgdaAny -> AgdaAny)
                     (AgdaAny -> AgdaAny) (AgdaAny -> AgdaAny)
-- Real.Real.R
d_R_28 :: T_Real_2 -> ()
d_R_28 = erased
-- Real.Real.fromℕ
d_fromℕ_30 :: T_Real_2 -> Integer -> AgdaAny
d_fromℕ_30 v0
  = case coe v0 of
      C_constructor_64 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 -> coe v2
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.∞ᵣ
d_'8734''7523'_32 :: T_Real_2 -> AgdaAny
d_'8734''7523'_32 v0
  = case coe v0 of
      C_constructor_64 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 -> coe v3
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real._+_
d__'43'__34 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'43'__34 v0
  = case coe v0 of
      C_constructor_64 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 -> coe v4
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real._*_
d__'42'__36 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'42'__36 v0
  = case coe v0 of
      C_constructor_64 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 -> coe v5
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real._∨_
d__'8744'__38 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'8744'__38 v0
  = case coe v0 of
      C_constructor_64 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 -> coe v6
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real._÷_
d__'247'__40 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'247'__40 v0
  = case coe v0 of
      C_constructor_64 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 -> coe v7
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.-_
d_'45'__42 :: T_Real_2 -> AgdaAny -> AgdaAny
d_'45'__42 v0
  = case coe v0 of
      C_constructor_64 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 -> coe v8
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.e^_
d_e'94'__44 :: T_Real_2 -> AgdaAny -> AgdaAny
d_e'94'__44 v0
  = case coe v0 of
      C_constructor_64 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 -> coe v9
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.√_
d_'8730'__46 :: T_Real_2 -> AgdaAny -> AgdaAny
d_'8730'__46 v0
  = case coe v0 of
      C_constructor_64 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 -> coe v10
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.I+
d_I'43'_48 :: T_Real_2 -> AgdaAny -> AgdaAny
d_I'43'_48 v0
  = case coe v0 of
      C_constructor_64 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 -> coe v11
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.log
d_log_50 :: T_Real_2 -> AgdaAny -> AgdaAny
d_log_50 v0
  = case coe v0 of
      C_constructor_64 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 -> coe v12
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.0ᵣ
d_0'7523'_52 :: T_Real_2 -> AgdaAny
d_0'7523'_52 v0 = coe d_fromℕ_30 v0 (0 :: Integer)
-- Real.Real.-∞ᵣ
d_'45''8734''7523'_54 :: T_Real_2 -> AgdaAny
d_'45''8734''7523'_54 v0
  = coe d_'45'__42 v0 (d_'8734''7523'_32 (coe v0))
-- Real.Real.logisticʳ
d_logistic'691'_56 :: T_Real_2 -> AgdaAny -> AgdaAny
d_logistic'691'_56 v0 v1
  = coe
      d__'247'__40 v0 (coe d_fromℕ_30 v0 (1 :: Integer))
      (coe
         d__'43'__34 v0 (coe d_fromℕ_30 v0 (1 :: Integer))
         (coe d_e'94'__44 v0 (coe d_'45'__42 v0 v1)))
-- Real.Real.1/_
d_1'47'__60 :: T_Real_2 -> AgdaAny -> AgdaAny
d_1'47'__60 v0
  = coe d__'247'__40 v0 (coe d_fromℕ_30 v0 (1 :: Integer))
-- Real.RealProp
d_RealProp_68 a0 = ()
newtype T_RealProp_68
  = C_constructor_286 (AgdaAny ->
                       AgdaAny -> MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20)
-- Real._._*_
d__'42'__74 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'42'__74 v0 = coe d__'42'__36 (coe v0)
-- Real._._+_
d__'43'__76 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'43'__76 v0 = coe d__'43'__34 (coe v0)
-- Real._._÷_
d__'247'__78 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'247'__78 v0 = coe d__'247'__40 (coe v0)
-- Real._._∨_
d__'8744'__80 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'8744'__80 v0 = coe d__'8744'__38 (coe v0)
-- Real._.-_
d_'45'__82 :: T_Real_2 -> AgdaAny -> AgdaAny
d_'45'__82 v0 = coe d_'45'__42 (coe v0)
-- Real._.-∞ᵣ
d_'45''8734''7523'_84 :: T_Real_2 -> AgdaAny
d_'45''8734''7523'_84 v0 = coe d_'45''8734''7523'_54 (coe v0)
-- Real._.0ᵣ
d_0'7523'_86 :: T_Real_2 -> AgdaAny
d_0'7523'_86 v0 = coe d_0'7523'_52 (coe v0)
-- Real._.1/_
d_1'47'__88 :: T_Real_2 -> AgdaAny -> AgdaAny
d_1'47'__88 v0 = coe d_1'47'__60 (coe v0)
-- Real._.I+
d_I'43'_90 :: T_Real_2 -> AgdaAny -> AgdaAny
d_I'43'_90 v0 = coe d_I'43'_48 (coe v0)
-- Real._.R
d_R_92 :: T_Real_2 -> ()
d_R_92 = erased
-- Real._.e^_
d_e'94'__94 :: T_Real_2 -> AgdaAny -> AgdaAny
d_e'94'__94 v0 = coe d_e'94'__44 (coe v0)
-- Real._.fromℕ
d_fromℕ_96 :: T_Real_2 -> Integer -> AgdaAny
d_fromℕ_96 v0 = coe d_fromℕ_30 (coe v0)
-- Real._.log
d_log_98 :: T_Real_2 -> AgdaAny -> AgdaAny
d_log_98 v0 = coe d_log_50 (coe v0)
-- Real._.logisticʳ
d_logistic'691'_100 :: T_Real_2 -> AgdaAny -> AgdaAny
d_logistic'691'_100 v0 = coe d_logistic'691'_56 (coe v0)
-- Real._.√_
d_'8730'__102 :: T_Real_2 -> AgdaAny -> AgdaAny
d_'8730'__102 v0 = coe d_'8730'__46 (coe v0)
-- Real._.∞ᵣ
d_'8734''7523'_104 :: T_Real_2 -> AgdaAny
d_'8734''7523'_104 v0 = coe d_'8734''7523'_32 (coe v0)
-- Real.RealProp._._*_
d__'42'__176 ::
  T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny -> AgdaAny
d__'42'__176 v0 ~v1 = du__'42'__176 v0
du__'42'__176 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'42'__176 v0 = coe d__'42'__36 (coe v0)
-- Real.RealProp._._+_
d__'43'__178 ::
  T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny -> AgdaAny
d__'43'__178 v0 ~v1 = du__'43'__178 v0
du__'43'__178 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'43'__178 v0 = coe d__'43'__34 (coe v0)
-- Real.RealProp._._÷_
d__'247'__180 ::
  T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny -> AgdaAny
d__'247'__180 v0 ~v1 = du__'247'__180 v0
du__'247'__180 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'247'__180 v0 = coe d__'247'__40 (coe v0)
-- Real.RealProp._._∨_
d__'8744'__182 ::
  T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny -> AgdaAny
d__'8744'__182 v0 ~v1 = du__'8744'__182 v0
du__'8744'__182 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'8744'__182 v0 = coe d__'8744'__38 (coe v0)
-- Real.RealProp._.-_
d_'45'__184 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_'45'__184 v0 ~v1 = du_'45'__184 v0
du_'45'__184 :: T_Real_2 -> AgdaAny -> AgdaAny
du_'45'__184 v0 = coe d_'45'__42 (coe v0)
-- Real.RealProp._.-∞ᵣ
d_'45''8734''7523'_186 :: T_Real_2 -> T_RealProp_68 -> AgdaAny
d_'45''8734''7523'_186 v0 ~v1 = du_'45''8734''7523'_186 v0
du_'45''8734''7523'_186 :: T_Real_2 -> AgdaAny
du_'45''8734''7523'_186 v0 = coe d_'45''8734''7523'_54 (coe v0)
-- Real.RealProp._.0ᵣ
d_0'7523'_188 :: T_Real_2 -> T_RealProp_68 -> AgdaAny
d_0'7523'_188 v0 ~v1 = du_0'7523'_188 v0
du_0'7523'_188 :: T_Real_2 -> AgdaAny
du_0'7523'_188 v0 = coe d_0'7523'_52 (coe v0)
-- Real.RealProp._.1/_
d_1'47'__190 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_1'47'__190 v0 ~v1 = du_1'47'__190 v0
du_1'47'__190 :: T_Real_2 -> AgdaAny -> AgdaAny
du_1'47'__190 v0 = coe d_1'47'__60 (coe v0)
-- Real.RealProp._.I+
d_I'43'_192 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_I'43'_192 v0 ~v1 = du_I'43'_192 v0
du_I'43'_192 :: T_Real_2 -> AgdaAny -> AgdaAny
du_I'43'_192 v0 = coe d_I'43'_48 (coe v0)
-- Real.RealProp._.R
d_R_194 :: T_Real_2 -> T_RealProp_68 -> ()
d_R_194 = erased
-- Real.RealProp._.e^_
d_e'94'__196 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_e'94'__196 v0 ~v1 = du_e'94'__196 v0
du_e'94'__196 :: T_Real_2 -> AgdaAny -> AgdaAny
du_e'94'__196 v0 = coe d_e'94'__44 (coe v0)
-- Real.RealProp._.fromℕ
d_fromℕ_198 :: T_Real_2 -> T_RealProp_68 -> Integer -> AgdaAny
d_fromℕ_198 v0 ~v1 = du_fromℕ_198 v0
du_fromℕ_198 :: T_Real_2 -> Integer -> AgdaAny
du_fromℕ_198 v0 = coe d_fromℕ_30 (coe v0)
-- Real.RealProp._.log
d_log_200 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_log_200 v0 ~v1 = du_log_200 v0
du_log_200 :: T_Real_2 -> AgdaAny -> AgdaAny
du_log_200 v0 = coe d_log_50 (coe v0)
-- Real.RealProp._.logisticʳ
d_logistic'691'_202 ::
  T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_logistic'691'_202 v0 ~v1 = du_logistic'691'_202 v0
du_logistic'691'_202 :: T_Real_2 -> AgdaAny -> AgdaAny
du_logistic'691'_202 v0 = coe d_logistic'691'_56 (coe v0)
-- Real.RealProp._.√_
d_'8730'__204 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_'8730'__204 v0 ~v1 = du_'8730'__204 v0
du_'8730'__204 :: T_Real_2 -> AgdaAny -> AgdaAny
du_'8730'__204 v0 = coe d_'8730'__46 (coe v0)
-- Real.RealProp._.∞ᵣ
d_'8734''7523'_206 :: T_Real_2 -> T_RealProp_68 -> AgdaAny
d_'8734''7523'_206 v0 ~v1 = du_'8734''7523'_206 v0
du_'8734''7523'_206 :: T_Real_2 -> AgdaAny
du_'8734''7523'_206 v0 = coe d_'8734''7523'_32 (coe v0)
-- Real.RealProp.+-neutˡ
d_'43''45'neut'737'_210 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'43''45'neut'737'_210 = erased
-- Real.RealProp.+-neutʳ
d_'43''45'neut'691'_214 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'43''45'neut'691'_214 = erased
-- Real.RealProp.*-neutˡ
d_'42''45'neut'737'_218 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''45'neut'737'_218 = erased
-- Real.RealProp.*-neutʳ
d_'42''45'neut'691'_222 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''45'neut'691'_222 = erased
-- Real.RealProp.*-nulˡ
d_'42''45'nul'737'_226 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''45'nul'737'_226 = erased
-- Real.RealProp.*-nulʳ
d_'42''45'nul'691'_230 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''45'nul'691'_230 = erased
-- Real.RealProp.minus-*-pushʳ
d_minus'45''42''45'push'691'_236 ::
  T_RealProp_68 ->
  AgdaAny ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_minus'45''42''45'push'691'_236 = erased
-- Real.RealProp.minus-invʳ
d_minus'45'inv'691'_240 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_minus'45'inv'691'_240 = erased
-- Real.RealProp.minus-idʳ
d_minus'45'id'691'_242 ::
  T_RealProp_68 -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_minus'45'id'691'_242 = erased
-- Real.RealProp.÷-nul
d_'247''45'nul_246 ::
  T_RealProp_68 ->
  AgdaAny ->
  (MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'247''45'nul_246 = erased
-- Real.RealProp.*-÷-cut
d_'42''45''247''45'cut_252 ::
  T_RealProp_68 ->
  AgdaAny ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''45''247''45'cut_252 = erased
-- Real.RealProp.fromℕ-inj
d_fromℕ'45'inj_258 ::
  T_RealProp_68 ->
  Integer ->
  Integer ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_fromℕ'45'inj_258 = erased
-- Real.RealProp.+-medial
d_'43''45'medial_268 ::
  T_RealProp_68 ->
  AgdaAny ->
  AgdaAny ->
  AgdaAny ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'43''45'medial_268 = erased
-- Real.RealProp._≡ᵣ?_
d__'8801''7523''63'__274 ::
  T_RealProp_68 ->
  AgdaAny ->
  AgdaAny -> MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d__'8801''7523''63'__274 v0
  = case coe v0 of
      C_constructor_286 v14 -> coe v14
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.RealProp._≤ᵣ?_
d__'8804''7523''63'__280 ::
  T_Real_2 ->
  T_RealProp_68 ->
  AgdaAny ->
  AgdaAny -> MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20
d__'8804''7523''63'__280 v0 v1 v2 v3
  = coe d__'8801''7523''63'__274 v1 v3 (coe d__'8744'__38 v0 v2 v3)
