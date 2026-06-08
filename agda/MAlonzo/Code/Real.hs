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

-- Real.Real
d_Real_2 = ()
data T_Real_2
  = C_constructor_58 (Integer -> AgdaAny)
                     (AgdaAny -> AgdaAny -> AgdaAny) (AgdaAny -> AgdaAny -> AgdaAny)
                     (AgdaAny -> AgdaAny -> AgdaAny) (AgdaAny -> AgdaAny -> AgdaAny)
                     (AgdaAny -> AgdaAny) (AgdaAny -> AgdaAny) (AgdaAny -> AgdaAny)
                     (AgdaAny -> AgdaAny) (AgdaAny -> AgdaAny)
-- Real.Real.R
d_R_26 :: T_Real_2 -> ()
d_R_26 = erased
-- Real.Real.fromℕ
d_fromℕ_28 :: T_Real_2 -> Integer -> AgdaAny
d_fromℕ_28 v0
  = case coe v0 of
      C_constructor_58 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 -> coe v2
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real._+_
d__'43'__30 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'43'__30 v0
  = case coe v0 of
      C_constructor_58 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 -> coe v3
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real._*_
d__'42'__32 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'42'__32 v0
  = case coe v0 of
      C_constructor_58 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 -> coe v4
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real._∨_
d__'8744'__34 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'8744'__34 v0
  = case coe v0 of
      C_constructor_58 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 -> coe v5
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real._÷_
d__'247'__36 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'247'__36 v0
  = case coe v0 of
      C_constructor_58 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 -> coe v6
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.-_
d_'45'__38 :: T_Real_2 -> AgdaAny -> AgdaAny
d_'45'__38 v0
  = case coe v0 of
      C_constructor_58 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 -> coe v7
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.e^_
d_e'94'__40 :: T_Real_2 -> AgdaAny -> AgdaAny
d_e'94'__40 v0
  = case coe v0 of
      C_constructor_58 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 -> coe v8
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.√_
d_'8730'__42 :: T_Real_2 -> AgdaAny -> AgdaAny
d_'8730'__42 v0
  = case coe v0 of
      C_constructor_58 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 -> coe v9
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.I+
d_I'43'_44 :: T_Real_2 -> AgdaAny -> AgdaAny
d_I'43'_44 v0
  = case coe v0 of
      C_constructor_58 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 -> coe v10
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.log
d_log_46 :: T_Real_2 -> AgdaAny -> AgdaAny
d_log_46 v0
  = case coe v0 of
      C_constructor_58 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 -> coe v11
      _ -> MAlonzo.RTE.mazUnreachableError
-- Real.Real.0ᵣ
d_0'7523'_48 :: T_Real_2 -> AgdaAny
d_0'7523'_48 v0 = coe d_fromℕ_28 v0 (0 :: Integer)
-- Real.Real.logisticʳ
d_logistic'691'_50 :: T_Real_2 -> AgdaAny -> AgdaAny
d_logistic'691'_50 v0 v1
  = coe
      d__'247'__36 v0 (coe d_fromℕ_28 v0 (1 :: Integer))
      (coe
         d__'43'__30 v0 (coe d_fromℕ_28 v0 (1 :: Integer))
         (coe d_e'94'__40 v0 (coe d_'45'__38 v0 v1)))
-- Real.Real.1/_
d_1'47'__54 :: T_Real_2 -> AgdaAny -> AgdaAny
d_1'47'__54 v0
  = coe d__'247'__36 v0 (coe d_fromℕ_28 v0 (1 :: Integer))
-- Real.RealProp
d_RealProp_62 a0 = ()
data T_RealProp_62 = C_constructor_158
-- Real._._*_
d__'42'__68 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'42'__68 v0 = coe d__'42'__32 (coe v0)
-- Real._._+_
d__'43'__70 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'43'__70 v0 = coe d__'43'__30 (coe v0)
-- Real._._÷_
d__'247'__72 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'247'__72 v0 = coe d__'247'__36 (coe v0)
-- Real._._∨_
d__'8744'__74 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
d__'8744'__74 v0 = coe d__'8744'__34 (coe v0)
-- Real._.-_
d_'45'__76 :: T_Real_2 -> AgdaAny -> AgdaAny
d_'45'__76 v0 = coe d_'45'__38 (coe v0)
-- Real._.0ᵣ
d_0'7523'_78 :: T_Real_2 -> AgdaAny
d_0'7523'_78 v0 = coe d_0'7523'_48 (coe v0)
-- Real._.1/_
d_1'47'__80 :: T_Real_2 -> AgdaAny -> AgdaAny
d_1'47'__80 v0 = coe d_1'47'__54 (coe v0)
-- Real._.I+
d_I'43'_82 :: T_Real_2 -> AgdaAny -> AgdaAny
d_I'43'_82 v0 = coe d_I'43'_44 (coe v0)
-- Real._.R
d_R_84 :: T_Real_2 -> ()
d_R_84 = erased
-- Real._.e^_
d_e'94'__86 :: T_Real_2 -> AgdaAny -> AgdaAny
d_e'94'__86 v0 = coe d_e'94'__40 (coe v0)
-- Real._.fromℕ
d_fromℕ_88 :: T_Real_2 -> Integer -> AgdaAny
d_fromℕ_88 v0 = coe d_fromℕ_28 (coe v0)
-- Real._.log
d_log_90 :: T_Real_2 -> AgdaAny -> AgdaAny
d_log_90 v0 = coe d_log_46 (coe v0)
-- Real._.logisticʳ
d_logistic'691'_92 :: T_Real_2 -> AgdaAny -> AgdaAny
d_logistic'691'_92 v0 = coe d_logistic'691'_50 (coe v0)
-- Real._.√_
d_'8730'__94 :: T_Real_2 -> AgdaAny -> AgdaAny
d_'8730'__94 v0 = coe d_'8730'__42 (coe v0)
-- Real.RealProp._._*_
d__'42'__114 ::
  T_Real_2 -> T_RealProp_62 -> AgdaAny -> AgdaAny -> AgdaAny
d__'42'__114 v0 ~v1 = du__'42'__114 v0
du__'42'__114 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'42'__114 v0 = coe d__'42'__32 (coe v0)
-- Real.RealProp._._+_
d__'43'__116 ::
  T_Real_2 -> T_RealProp_62 -> AgdaAny -> AgdaAny -> AgdaAny
d__'43'__116 v0 ~v1 = du__'43'__116 v0
du__'43'__116 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'43'__116 v0 = coe d__'43'__30 (coe v0)
-- Real.RealProp._._÷_
d__'247'__118 ::
  T_Real_2 -> T_RealProp_62 -> AgdaAny -> AgdaAny -> AgdaAny
d__'247'__118 v0 ~v1 = du__'247'__118 v0
du__'247'__118 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'247'__118 v0 = coe d__'247'__36 (coe v0)
-- Real.RealProp._._∨_
d__'8744'__120 ::
  T_Real_2 -> T_RealProp_62 -> AgdaAny -> AgdaAny -> AgdaAny
d__'8744'__120 v0 ~v1 = du__'8744'__120 v0
du__'8744'__120 :: T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'8744'__120 v0 = coe d__'8744'__34 (coe v0)
-- Real.RealProp._.-_
d_'45'__122 :: T_Real_2 -> T_RealProp_62 -> AgdaAny -> AgdaAny
d_'45'__122 v0 ~v1 = du_'45'__122 v0
du_'45'__122 :: T_Real_2 -> AgdaAny -> AgdaAny
du_'45'__122 v0 = coe d_'45'__38 (coe v0)
-- Real.RealProp._.0ᵣ
d_0'7523'_124 :: T_Real_2 -> T_RealProp_62 -> AgdaAny
d_0'7523'_124 v0 ~v1 = du_0'7523'_124 v0
du_0'7523'_124 :: T_Real_2 -> AgdaAny
du_0'7523'_124 v0 = coe d_0'7523'_48 (coe v0)
-- Real.RealProp._.1/_
d_1'47'__126 :: T_Real_2 -> T_RealProp_62 -> AgdaAny -> AgdaAny
d_1'47'__126 v0 ~v1 = du_1'47'__126 v0
du_1'47'__126 :: T_Real_2 -> AgdaAny -> AgdaAny
du_1'47'__126 v0 = coe d_1'47'__54 (coe v0)
-- Real.RealProp._.I+
d_I'43'_128 :: T_Real_2 -> T_RealProp_62 -> AgdaAny -> AgdaAny
d_I'43'_128 v0 ~v1 = du_I'43'_128 v0
du_I'43'_128 :: T_Real_2 -> AgdaAny -> AgdaAny
du_I'43'_128 v0 = coe d_I'43'_44 (coe v0)
-- Real.RealProp._.R
d_R_130 :: T_Real_2 -> T_RealProp_62 -> ()
d_R_130 = erased
-- Real.RealProp._.e^_
d_e'94'__132 :: T_Real_2 -> T_RealProp_62 -> AgdaAny -> AgdaAny
d_e'94'__132 v0 ~v1 = du_e'94'__132 v0
du_e'94'__132 :: T_Real_2 -> AgdaAny -> AgdaAny
du_e'94'__132 v0 = coe d_e'94'__40 (coe v0)
-- Real.RealProp._.fromℕ
d_fromℕ_134 :: T_Real_2 -> T_RealProp_62 -> Integer -> AgdaAny
d_fromℕ_134 v0 ~v1 = du_fromℕ_134 v0
du_fromℕ_134 :: T_Real_2 -> Integer -> AgdaAny
du_fromℕ_134 v0 = coe d_fromℕ_28 (coe v0)
-- Real.RealProp._.log
d_log_136 :: T_Real_2 -> T_RealProp_62 -> AgdaAny -> AgdaAny
d_log_136 v0 ~v1 = du_log_136 v0
du_log_136 :: T_Real_2 -> AgdaAny -> AgdaAny
du_log_136 v0 = coe d_log_46 (coe v0)
-- Real.RealProp._.logisticʳ
d_logistic'691'_138 ::
  T_Real_2 -> T_RealProp_62 -> AgdaAny -> AgdaAny
d_logistic'691'_138 v0 ~v1 = du_logistic'691'_138 v0
du_logistic'691'_138 :: T_Real_2 -> AgdaAny -> AgdaAny
du_logistic'691'_138 v0 = coe d_logistic'691'_50 (coe v0)
-- Real.RealProp._.√_
d_'8730'__140 :: T_Real_2 -> T_RealProp_62 -> AgdaAny -> AgdaAny
d_'8730'__140 v0 ~v1 = du_'8730'__140 v0
du_'8730'__140 :: T_Real_2 -> AgdaAny -> AgdaAny
du_'8730'__140 v0 = coe d_'8730'__42 (coe v0)
-- Real.RealProp.+-neutˡ
d_'43''45'neut'737'_144 ::
  T_RealProp_62 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'43''45'neut'737'_144 = erased
-- Real.RealProp.+-neutʳ
d_'43''45'neut'691'_148 ::
  T_RealProp_62 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'43''45'neut'691'_148 = erased
-- Real.RealProp.*-neutˡ
d_'42''45'neut'737'_152 ::
  T_RealProp_62 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''45'neut'737'_152 = erased
-- Real.RealProp.*-neutʳ
d_'42''45'neut'691'_156 ::
  T_RealProp_62 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'42''45'neut'691'_156 = erased
