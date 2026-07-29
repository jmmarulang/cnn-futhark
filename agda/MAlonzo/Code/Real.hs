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
data T_RealProp_68 = C_constructor_228
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
d__'42'__152 ::
  T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny -> AgdaAny
d__'42'__152 v0 ~v1 = du__'42'__152 v0
du__'42'__152 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'42'__152 v0 = coe d__'42'__36 (coe v0)
-- Real.RealProp._._+_
d__'43'__154 ::
  T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny -> AgdaAny
d__'43'__154 v0 ~v1 = du__'43'__154 v0
du__'43'__154 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'43'__154 v0 = coe d__'43'__34 (coe v0)
-- Real.RealProp._._÷_
d__'247'__156 ::
  T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny -> AgdaAny
d__'247'__156 v0 ~v1 = du__'247'__156 v0
du__'247'__156 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'247'__156 v0 = coe d__'247'__40 (coe v0)
-- Real.RealProp._._∨_
d__'8744'__158 ::
  T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny -> AgdaAny
d__'8744'__158 v0 ~v1 = du__'8744'__158 v0
du__'8744'__158 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'8744'__158 v0 = coe d__'8744'__38 (coe v0)
-- Real.RealProp._.-_
d_'45'__160 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_'45'__160 v0 ~v1 = du_'45'__160 v0
du_'45'__160 :: T_Real_2 -> AgdaAny -> AgdaAny
du_'45'__160 v0 = coe d_'45'__42 (coe v0)
-- Real.RealProp._.-∞ᵣ
d_'45''8734''7523'_162 :: T_Real_2 -> T_RealProp_68 -> AgdaAny
d_'45''8734''7523'_162 v0 ~v1 = du_'45''8734''7523'_162 v0
du_'45''8734''7523'_162 :: T_Real_2 -> AgdaAny
du_'45''8734''7523'_162 v0 = coe d_'45''8734''7523'_54 (coe v0)
-- Real.RealProp._.0ᵣ
d_0'7523'_164 :: T_Real_2 -> T_RealProp_68 -> AgdaAny
d_0'7523'_164 v0 ~v1 = du_0'7523'_164 v0
du_0'7523'_164 :: T_Real_2 -> AgdaAny
du_0'7523'_164 v0 = coe d_0'7523'_52 (coe v0)
-- Real.RealProp._.1/_
d_1'47'__166 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_1'47'__166 v0 ~v1 = du_1'47'__166 v0
du_1'47'__166 :: T_Real_2 -> AgdaAny -> AgdaAny
du_1'47'__166 v0 = coe d_1'47'__60 (coe v0)
-- Real.RealProp._.I+
d_I'43'_168 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_I'43'_168 v0 ~v1 = du_I'43'_168 v0
du_I'43'_168 :: T_Real_2 -> AgdaAny -> AgdaAny
du_I'43'_168 v0 = coe d_I'43'_48 (coe v0)
-- Real.RealProp._.R
d_R_170 :: T_Real_2 -> T_RealProp_68 -> ()
d_R_170 = erased
-- Real.RealProp._.e^_
d_e'94'__172 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_e'94'__172 v0 ~v1 = du_e'94'__172 v0
du_e'94'__172 :: T_Real_2 -> AgdaAny -> AgdaAny
du_e'94'__172 v0 = coe d_e'94'__44 (coe v0)
-- Real.RealProp._.fromℕ
d_fromℕ_174 :: T_Real_2 -> T_RealProp_68 -> Integer -> AgdaAny
d_fromℕ_174 v0 ~v1 = du_fromℕ_174 v0
du_fromℕ_174 :: T_Real_2 -> Integer -> AgdaAny
du_fromℕ_174 v0 = coe d_fromℕ_30 (coe v0)
-- Real.RealProp._.log
d_log_176 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_log_176 v0 ~v1 = du_log_176 v0
du_log_176 :: T_Real_2 -> AgdaAny -> AgdaAny
du_log_176 v0 = coe d_log_50 (coe v0)
-- Real.RealProp._.logisticʳ
d_logistic'691'_178 ::
  T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_logistic'691'_178 v0 ~v1 = du_logistic'691'_178 v0
du_logistic'691'_178 :: T_Real_2 -> AgdaAny -> AgdaAny
du_logistic'691'_178 v0 = coe d_logistic'691'_56 (coe v0)
-- Real.RealProp._.√_
d_'8730'__180 :: T_Real_2 -> T_RealProp_68 -> AgdaAny -> AgdaAny
d_'8730'__180 v0 ~v1 = du_'8730'__180 v0
du_'8730'__180 :: T_Real_2 -> AgdaAny -> AgdaAny
du_'8730'__180 v0 = coe d_'8730'__46 (coe v0)
-- Real.RealProp._.∞ᵣ
d_'8734''7523'_182 :: T_Real_2 -> T_RealProp_68 -> AgdaAny
d_'8734''7523'_182 v0 ~v1 = du_'8734''7523'_182 v0
du_'8734''7523'_182 :: T_Real_2 -> AgdaAny
du_'8734''7523'_182 v0 = coe d_'8734''7523'_32 (coe v0)
-- Real.RealProp.+-neutˡ
d_'43''45'neut'737'_186 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'43''45'neut'737'_186 = erased
-- Real.RealProp.+-neutʳ
d_'43''45'neut'691'_190 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'43''45'neut'691'_190 = erased
-- Real.RealProp.*-neutˡ
d_'42''45'neut'737'_194 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''45'neut'737'_194 = erased
-- Real.RealProp.*-neutʳ
d_'42''45'neut'691'_198 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''45'neut'691'_198 = erased
-- Real.RealProp.*-nulˡ
d_'42''45'nul'737'_202 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''45'nul'737'_202 = erased
-- Real.RealProp.*-nulʳ
d_'42''45'nul'691'_206 ::
  T_RealProp_68 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''45'nul'691'_206 = erased
-- Real.RealProp.÷-nul
d_'247''45'nul_210 ::
  T_RealProp_68 ->
  AgdaAny ->
  (MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'247''45'nul_210 = erased
-- Real.RealProp.fromℕ-inj
d_fromℕ'45'inj_216 ::
  T_RealProp_68 ->
  Integer ->
  Integer ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_fromℕ'45'inj_216 = erased
-- Real.RealProp.+-medial
d_'43''45'medial_226 ::
  T_RealProp_68 ->
  AgdaAny ->
  AgdaAny ->
  AgdaAny ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'43''45'medial_226 = erased
