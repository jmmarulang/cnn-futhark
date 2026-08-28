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

module MAlonzo.Code.Grad where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.Equality
import qualified MAlonzo.Code.Agda.Builtin.Maybe
import qualified MAlonzo.Code.Agda.Builtin.Sigma
import qualified MAlonzo.Code.Ar
import qualified MAlonzo.Code.Data.Nat.Base
import qualified MAlonzo.Code.Data.Nat.Properties
import qualified MAlonzo.Code.Lang

-- Grad._.Tel
d_Tel_6 a0 a1 = ()
data T_Tel_6
  = C_ε_8 | C__'9657'__10 T_Tel_6 MAlonzo.Code.Lang.T_E_214
-- Grad._.Env
d_Env_12 a0 a1 = ()
data T_Env_12
  = C_ε_14 | C_skip_16 T_Env_12 |
    C__'9657'__18 T_Env_12 MAlonzo.Code.Lang.T_E_214
-- Grad._.EE
d_EE_20 a0 a1 = ()
data T_EE_20
  = C_env_22 T_Env_12 |
    C_let'8242'_24 [Integer] MAlonzo.Code.Lang.T_E_214 T_EE_20
-- Grad._.env-wk
d_env'45'wk_26 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8838'__352 -> T_Env_12 -> T_Env_12
d_env'45'wk_26 v0 v1 v2 v3 v4
  = case coe v4 of
      C_ε_14 -> coe C_ε_14
      C_skip_16 v8
        -> case coe v2 of
             MAlonzo.Code.Lang.C__'9657'__40 v9 v10
               -> coe
                    C_skip_16
                    (d_env'45'wk_26 (coe v0) (coe v1) (coe v9) (coe v3) (coe v8))
             _ -> MAlonzo.RTE.mazUnreachableError
      C__'9657'__18 v8 v9
        -> case coe v2 of
             MAlonzo.Code.Lang.C__'9657'__40 v10 v11
               -> coe
                    C__'9657'__18
                    (d_env'45'wk_26 (coe v0) (coe v1) (coe v10) (coe v3) (coe v8))
                    (MAlonzo.Code.Lang.d_wk_372
                       (coe v0) (coe v1) (coe v11) (coe v3) (coe v9))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.ee-wk
d_ee'45'wk_40 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8838'__352 -> T_EE_20 -> T_EE_20
d_ee'45'wk_40 v0 v1 v2 v3 v4
  = case coe v4 of
      C_env_22 v7
        -> coe
             C_env_22
             (d_env'45'wk_26 (coe v0) (coe v1) (coe v2) (coe v3) (coe v7))
      C_let'8242'_24 v6 v8 v9
        -> coe
             C_let'8242'_24 v6
             (MAlonzo.Code.Lang.d_wk_372
                (coe v0) (coe v1) (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) (coe v3)
                (coe v8))
             (d_ee'45'wk_40
                (coe
                   MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)))
                (coe
                   MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)))
                (coe v2) (coe MAlonzo.Code.Lang.C_keep_358 v3) (coe v9))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.ee-tail
d_ee'45'tail_52 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> T_EE_20 -> T_EE_20
d_ee'45'tail_52 ~v0 ~v1 ~v2 v3 = du_ee'45'tail_52 v3
du_ee'45'tail_52 :: T_EE_20 -> T_EE_20
du_ee'45'tail_52 v0
  = case coe v0 of
      C_env_22 v3
        -> case coe v3 of
             C_skip_16 v7 -> coe C_env_22 v7
             C__'9657'__18 v7 v8 -> coe C_env_22 v7
             _ -> MAlonzo.RTE.mazUnreachableError
      C_let'8242'_24 v2 v4 v5
        -> coe C_let'8242'_24 v2 v4 (coe du_ee'45'tail_52 (coe v5))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.env-wk-zero
d_env'45'wk'45'zero_62 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_Env_12 -> MAlonzo.Code.Lang.T__'8838'__352 -> T_Env_12
d_env'45'wk'45'zero_62 v0 ~v1 v2 v3 v4
  = du_env'45'wk'45'zero_62 v0 v2 v3 v4
du_env'45'wk'45'zero_62 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_Env_12 -> MAlonzo.Code.Lang.T__'8838'__352 -> T_Env_12
du_env'45'wk'45'zero_62 v0 v1 v2 v3
  = case coe v3 of
      MAlonzo.Code.Lang.C_ε_354 -> coe v2
      MAlonzo.Code.Lang.C_skip_356 v7
        -> case coe v1 of
             MAlonzo.Code.Lang.C__'9657'__40 v8 v9
               -> case coe v9 of
                    MAlonzo.Code.Lang.C_ix_32 v10
                      -> coe
                           C_skip_16
                           (coe du_env'45'wk'45'zero_62 (coe v0) (coe v8) (coe v2) (coe v7))
                    MAlonzo.Code.Lang.C_ar_34 v10
                      -> coe
                           C__'9657'__18
                           (coe du_env'45'wk'45'zero_62 (coe v0) (coe v8) (coe v2) (coe v7))
                           (coe MAlonzo.Code.Lang.C_zero_218)
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_keep_358 v7
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v8 v9
               -> case coe v1 of
                    MAlonzo.Code.Lang.C__'9657'__40 v10 v11
                      -> case coe v2 of
                           C_skip_16 v15
                             -> coe
                                  C_skip_16
                                  (coe
                                     du_env'45'wk'45'zero_62 (coe v8) (coe v10) (coe v15) (coe v7))
                           C__'9657'__18 v15 v16
                             -> coe
                                  C__'9657'__18
                                  (coe
                                     du_env'45'wk'45'zero_62 (coe v8) (coe v10) (coe v15) (coe v7))
                                  v16
                           _ -> MAlonzo.RTE.mazUnreachableError
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.ee-wk-zero
d_ee'45'wk'45'zero_92 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_EE_20 -> MAlonzo.Code.Lang.T__'8838'__352 -> T_EE_20
d_ee'45'wk'45'zero_92 v0 ~v1 v2 v3 v4
  = du_ee'45'wk'45'zero_92 v0 v2 v3 v4
du_ee'45'wk'45'zero_92 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_EE_20 -> MAlonzo.Code.Lang.T__'8838'__352 -> T_EE_20
du_ee'45'wk'45'zero_92 v0 v1 v2 v3
  = case coe v2 of
      C_env_22 v6
        -> coe
             C_env_22
             (coe du_env'45'wk'45'zero_62 (coe v0) (coe v1) (coe v6) (coe v3))
      C_let'8242'_24 v5 v7 v8
        -> coe
             C_let'8242'_24 v5 v7
             (coe du_ee'45'wk'45'zero_92 (coe v0) (coe v1) (coe v8) (coe v3))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.ee-push-zero
d_ee'45'push'45'zero_104 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> [Integer] -> T_EE_20 -> T_EE_20
d_ee'45'push'45'zero_104 v0 ~v1 v2 v3
  = du_ee'45'push'45'zero_104 v0 v2 v3
du_ee'45'push'45'zero_104 ::
  MAlonzo.Code.Lang.T_Ctx_36 -> [Integer] -> T_EE_20 -> T_EE_20
du_ee'45'push'45'zero_104 v0 v1 v2
  = coe
      du_ee'45'wk'45'zero_92 (coe v0)
      (coe
         MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
         (coe MAlonzo.Code.Lang.C_ar_34 (coe v1)))
      (coe v2)
      (coe
         MAlonzo.Code.Lang.C_skip_356
         (MAlonzo.Code.Lang.d_'8838''45'eq_494 (coe v0)))
-- Grad._.zero-env
d_zero'45'env_108 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> T_Env_12
d_zero'45'env_108 v0 ~v1 = du_zero'45'env_108 v0
du_zero'45'env_108 :: MAlonzo.Code.Lang.T_Ctx_36 -> T_Env_12
du_zero'45'env_108 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_ε_38 -> coe C_ε_14
      MAlonzo.Code.Lang.C__'9657'__40 v1 v2
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ix_32 v3
               -> coe C_skip_16 (coe du_zero'45'env_108 (coe v1))
             MAlonzo.Code.Lang.C_ar_34 v3
               -> coe
                    C__'9657'__18 (coe du_zero'45'env_108 (coe v1))
                    (coe MAlonzo.Code.Lang.C_zero_218)
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.zero-ee
d_zero'45'ee_118 ::
  MAlonzo.Code.Lang.T_Ctx_36 -> MAlonzo.Code.Lang.T_Ctx_36 -> T_EE_20
d_zero'45'ee_118 v0 ~v1 = du_zero'45'ee_118 v0
du_zero'45'ee_118 :: MAlonzo.Code.Lang.T_Ctx_36 -> T_EE_20
du_zero'45'ee_118 v0
  = coe C_env_22 (coe du_zero'45'env_108 (coe v0))
-- Grad._.env-update+
d_env'45'update'43'_124 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  T_Env_12 ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  MAlonzo.Code.Lang.T_E_214 -> T_Env_12
d_env'45'update'43'_124 v0 ~v1 ~v2 v3 v4 v5
  = du_env'45'update'43'_124 v0 v3 v4 v5
du_env'45'update'43'_124 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_Env_12 ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  MAlonzo.Code.Lang.T_E_214 -> T_Env_12
du_env'45'update'43'_124 v0 v1 v2 v3
  = case coe v1 of
      C_skip_16 v7
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v8 v9
               -> case coe v2 of
                    MAlonzo.Code.Lang.C_there_62 v13
                      -> coe
                           C_skip_16
                           (coe du_env'45'update'43'_124 (coe v8) (coe v7) (coe v13) (coe v3))
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      C__'9657'__18 v7 v8
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v9 v10
               -> case coe v2 of
                    MAlonzo.Code.Lang.C_here_60
                      -> coe
                           C__'9657'__18 v7
                           (coe
                              MAlonzo.Code.Lang.C_bin_242 (coe MAlonzo.Code.Lang.C_plus_190) v8
                              v3)
                    MAlonzo.Code.Lang.C_there_62 v14
                      -> coe
                           C__'9657'__18
                           (coe du_env'45'update'43'_124 (coe v9) (coe v7) (coe v14) (coe v3))
                           v8
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.ee-update+
d_ee'45'update'43'_150 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  T_EE_20 ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  MAlonzo.Code.Lang.T_E_214 -> T_EE_20
d_ee'45'update'43'_150 v0 v1 v2 v3 v4 v5
  = case coe v3 of
      C_env_22 v8
        -> coe
             C_env_22
             (coe du_env'45'update'43'_124 (coe v0) (coe v8) (coe v4) (coe v5))
      C_let'8242'_24 v7 v9 v10
        -> coe
             C_let'8242'_24 v7 v9
             (d_ee'45'update'43'_150
                (coe v0)
                (coe
                   MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v7)))
                (coe v2) (coe v10) (coe v4)
                (coe
                   MAlonzo.Code.Lang.d__'8593'_512 v1
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v2))
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v7)) v5))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.env-map-sum
d_env'45'map'45'sum_166 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> [Integer] -> T_Env_12 -> T_Env_12
d_env'45'map'45'sum_166 v0 ~v1 v2 v3
  = du_env'45'map'45'sum_166 v0 v2 v3
du_env'45'map'45'sum_166 ::
  MAlonzo.Code.Lang.T_Ctx_36 -> [Integer] -> T_Env_12 -> T_Env_12
du_env'45'map'45'sum_166 v0 v1 v2
  = case coe v2 of
      C_ε_14 -> coe C_ε_14
      C_skip_16 v6
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v7 v8
               -> coe
                    C_skip_16 (coe du_env'45'map'45'sum_166 (coe v7) (coe v1) (coe v6))
             _ -> MAlonzo.RTE.mazUnreachableError
      C__'9657'__18 v6 v7
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v8 v9
               -> coe
                    C__'9657'__18
                    (coe du_env'45'map'45'sum_166 (coe v8) (coe v1) (coe v6))
                    (coe MAlonzo.Code.Lang.C_sum_234 v1 v7)
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.ee-fold
d_ee'45'fold_174 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> T_EE_20 -> T_Env_12
d_ee'45'fold_174 v0 v1 v2
  = case coe v2 of
      C_env_22 v5 -> coe v5
      C_let'8242'_24 v4 v6 v7
        -> coe
             du_map'45'let_192 (coe v1) (coe v4) (coe v6) (coe v0)
             (coe
                d_ee'45'fold_174 (coe v0)
                (coe
                   MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v4)))
                (coe v7))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._._.map-let
d_map'45'let_192 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  T_EE_20 -> MAlonzo.Code.Lang.T_Ctx_36 -> T_Env_12 -> T_Env_12
d_map'45'let_192 ~v0 v1 v2 v3 ~v4 v5 v6
  = du_map'45'let_192 v1 v2 v3 v5 v6
du_map'45'let_192 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> T_Env_12 -> T_Env_12
du_map'45'let_192 v0 v1 v2 v3 v4
  = case coe v4 of
      C_ε_14 -> coe C_ε_14
      C_skip_16 v8
        -> case coe v3 of
             MAlonzo.Code.Lang.C__'9657'__40 v9 v10
               -> coe
                    C_skip_16
                    (coe
                       du_map'45'let_192 (coe v0) (coe v1) (coe v2) (coe v9) (coe v8))
             _ -> MAlonzo.RTE.mazUnreachableError
      C__'9657'__18 v8 v9
        -> case coe v3 of
             MAlonzo.Code.Lang.C__'9657'__40 v10 v11
               -> let v12
                        = MAlonzo.Code.Lang.d_stren'45''8707'_876
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                               (coe MAlonzo.Code.Lang.C_ar_34 (coe v1)))
                            (coe v11) (coe MAlonzo.Code.Lang.C_ar_34 (coe v1)) (coe v9)
                            (coe MAlonzo.Code.Lang.C_here_60) in
                  coe
                    (case coe v12 of
                       MAlonzo.Code.Agda.Builtin.Maybe.C_just_16 v13
                         -> case coe v13 of
                              MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                -> coe
                                     C__'9657'__18
                                     (coe
                                        du_map'45'let_192 (coe v0) (coe v1) (coe v2) (coe v10)
                                        (coe v8))
                                     v14
                              _ -> MAlonzo.RTE.mazUnreachableError
                       MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                         -> coe
                              C__'9657'__18
                              (coe
                                 du_map'45'let_192 (coe v0) (coe v1) (coe v2) (coe v10) (coe v8))
                              (coe MAlonzo.Code.Lang.C_let'8242'_246 v1 v2 v9)
                       _ -> MAlonzo.RTE.mazUnreachableError)
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.env-plus
d_env'45'plus_218 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> T_Env_12 -> T_Env_12 -> T_Env_12
d_env'45'plus_218 v0 ~v1 v2 v3 = du_env'45'plus_218 v0 v2 v3
du_env'45'plus_218 ::
  MAlonzo.Code.Lang.T_Ctx_36 -> T_Env_12 -> T_Env_12 -> T_Env_12
du_env'45'plus_218 v0 v1 v2
  = case coe v1 of
      C_ε_14 -> coe v2
      C_skip_16 v6
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v7 v8
               -> case coe v2 of
                    C_skip_16 v12
                      -> coe
                           C_skip_16 (coe du_env'45'plus_218 (coe v7) (coe v6) (coe v12))
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      C__'9657'__18 v6 v7
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v8 v9
               -> case coe v2 of
                    C__'9657'__18 v13 v14
                      -> coe
                           C__'9657'__18 (coe du_env'45'plus_218 (coe v8) (coe v6) (coe v13))
                           (coe
                              MAlonzo.Code.Lang.C_bin_242 (coe MAlonzo.Code.Lang.C_plus_190) v7
                              v14)
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.ee-plus
d_ee'45'plus_238 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> T_EE_20 -> T_EE_20 -> T_EE_20
d_ee'45'plus_238 v0 v1 v2 v3
  = case coe v2 of
      C_env_22 v6
        -> case coe v3 of
             C_env_22 v9
               -> coe C_env_22 (coe du_env'45'plus_218 (coe v0) (coe v6) (coe v9))
             C_let'8242'_24 v8 v10 v11
               -> coe
                    C_let'8242'_24 v8 v10
                    (d_ee'45'plus_238
                       (coe v0)
                       (coe
                          MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                          (coe MAlonzo.Code.Lang.C_ar_34 (coe v8)))
                       (coe
                          d_ee'45'wk_40 (coe v1)
                          (coe
                             MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                             (coe MAlonzo.Code.Lang.C_ar_34 (coe v8)))
                          (coe v0)
                          (coe
                             MAlonzo.Code.Lang.C_skip_356
                             (MAlonzo.Code.Lang.d_'8838''45'eq_494 (coe v1)))
                          (coe C_env_22 v6))
                       (coe v11))
             _ -> MAlonzo.RTE.mazUnreachableError
      C_let'8242'_24 v5 v7 v8
        -> coe
             C_let'8242'_24 v5 v7
             (d_ee'45'plus_238
                (coe v0)
                (coe
                   MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)))
                (coe v8)
                (coe
                   d_ee'45'wk_40 (coe v1)
                   (coe
                      MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                      (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)))
                   (coe v0)
                   (coe
                      MAlonzo.Code.Lang.C_skip_356
                      (MAlonzo.Code.Lang.d_'8838''45'eq_494 (coe v1)))
                   (coe v3)))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.let-depth
d_let'45'depth_256 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> T_EE_20 -> Integer
d_let'45'depth_256 ~v0 ~v1 v2 = du_let'45'depth_256 v2
du_let'45'depth_256 :: T_EE_20 -> Integer
du_let'45'depth_256 v0
  = case coe v0 of
      C_env_22 v3 -> coe (0 :: Integer)
      C_let'8242'_24 v2 v4 v5
        -> coe
             addInt (coe (1 :: Integer)) (coe du_let'45'depth_256 (coe v5))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.ee-wk-depth
d_ee'45'wk'45'depth_268 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_EE_20 ->
  MAlonzo.Code.Lang.T__'8838'__352 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_ee'45'wk'45'depth_268 = erased
-- Grad._.sub-<₁
d_sub'45''60''8321'_286 ::
  Integer ->
  Integer ->
  Integer ->
  MAlonzo.Code.Data.Nat.Base.T__'8804'__22 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
  MAlonzo.Code.Data.Nat.Base.T__'8804'__22
d_sub'45''60''8321'_286 ~v0 ~v1 ~v2 v3 ~v4
  = du_sub'45''60''8321'_286 v3
du_sub'45''60''8321'_286 ::
  MAlonzo.Code.Data.Nat.Base.T__'8804'__22 ->
  MAlonzo.Code.Data.Nat.Base.T__'8804'__22
du_sub'45''60''8321'_286 v0 = coe v0
-- Grad._.eep
d_eep_296 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_EE_20 ->
  T_EE_20 ->
  Integer -> MAlonzo.Code.Data.Nat.Base.T__'8804'__22 -> T_EE_20
d_eep_296 v0 v1 v2 v3 ~v4 v5 = du_eep_296 v0 v1 v2 v3 v5
du_eep_296 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_EE_20 ->
  T_EE_20 -> MAlonzo.Code.Data.Nat.Base.T__'8804'__22 -> T_EE_20
du_eep_296 v0 v1 v2 v3 v4
  = case coe v2 of
      C_env_22 v7
        -> case coe v3 of
             C_env_22 v10
               -> coe
                    C_env_22 (coe du_env'45'plus_218 (coe v0) (coe v7) (coe v10))
             C_let'8242'_24 v9 v11 v12
               -> case coe v4 of
                    MAlonzo.Code.Data.Nat.Base.C_s'8804's_34 v15
                      -> coe
                           C_let'8242'_24 v9 v11
                           (coe
                              du_eep_296 (coe v0)
                              (coe
                                 MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                 (coe MAlonzo.Code.Lang.C_ar_34 (coe v9)))
                              (coe
                                 d_ee'45'wk_40 (coe v1)
                                 (coe
                                    MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                    (coe MAlonzo.Code.Lang.C_ar_34 (coe v9)))
                                 (coe v0)
                                 (coe
                                    MAlonzo.Code.Lang.C_skip_356
                                    (MAlonzo.Code.Lang.d_'8838''45'eq_494 (coe v1)))
                                 (coe C_env_22 v7))
                              (coe v12) (coe v15))
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      C_let'8242'_24 v6 v8 v9
        -> case coe v4 of
             MAlonzo.Code.Data.Nat.Base.C_s'8804's_34 v12
               -> coe
                    C_let'8242'_24 v6 v8
                    (coe
                       du_eep_296 (coe v0)
                       (coe
                          MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                          (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)))
                       (coe v9)
                       (coe
                          d_ee'45'wk_40 (coe v1)
                          (coe
                             MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                             (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)))
                          (coe v0)
                          (coe
                             MAlonzo.Code.Lang.C_skip_356
                             (MAlonzo.Code.Lang.d_'8838''45'eq_494 (coe v1)))
                          (coe v3))
                       (coe v12))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.ee-plus′
d_ee'45'plus'8242'_332 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> T_EE_20 -> T_EE_20 -> T_EE_20
d_ee'45'plus'8242'_332 v0 v1 v2 v3
  = coe
      du_eep_296 (coe v0) (coe v1) (coe v2) (coe v3)
      (coe
         MAlonzo.Code.Data.Nat.Properties.d_'8804''45'refl_2900
         (coe
            addInt
            (coe
               addInt (coe (1 :: Integer)) (coe du_let'45'depth_256 (coe v2)))
            (coe du_let'45'depth_256 (coe v3))))
-- Grad._.env-lookup
d_env'45'lookup_338 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  T_Env_12 ->
  MAlonzo.Code.Lang.T__'8712'__58 -> MAlonzo.Code.Lang.T_E_214
d_env'45'lookup_338 v0 ~v1 ~v2 v3 v4
  = du_env'45'lookup_338 v0 v3 v4
du_env'45'lookup_338 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_Env_12 ->
  MAlonzo.Code.Lang.T__'8712'__58 -> MAlonzo.Code.Lang.T_E_214
du_env'45'lookup_338 v0 v1 v2
  = case coe v1 of
      C_skip_16 v6
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v7 v8
               -> case coe v2 of
                    MAlonzo.Code.Lang.C_there_62 v12
                      -> coe du_env'45'lookup_338 (coe v7) (coe v6) (coe v12)
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      C__'9657'__18 v6 v7
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v8 v9
               -> case coe v2 of
                    MAlonzo.Code.Lang.C_here_60 -> coe v7
                    MAlonzo.Code.Lang.C_there_62 v13
                      -> coe du_env'45'lookup_338 (coe v8) (coe v6) (coe v13)
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.env-rm-/
d_env'45'rm'45''47'_356 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  T_Env_12 -> MAlonzo.Code.Lang.T__'8712'__58 -> T_Env_12
d_env'45'rm'45''47'_356 v0 ~v1 ~v2 v3 v4
  = du_env'45'rm'45''47'_356 v0 v3 v4
du_env'45'rm'45''47'_356 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_Env_12 -> MAlonzo.Code.Lang.T__'8712'__58 -> T_Env_12
du_env'45'rm'45''47'_356 v0 v1 v2
  = case coe v1 of
      C_skip_16 v6
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v7 v8
               -> case coe v2 of
                    MAlonzo.Code.Lang.C_there_62 v12
                      -> coe
                           C_skip_16
                           (coe du_env'45'rm'45''47'_356 (coe v7) (coe v6) (coe v12))
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      C__'9657'__18 v6 v7
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v8 v9
               -> case coe v2 of
                    MAlonzo.Code.Lang.C_here_60 -> coe v6
                    MAlonzo.Code.Lang.C_there_62 v13
                      -> coe
                           C__'9657'__18
                           (coe du_env'45'rm'45''47'_356 (coe v8) (coe v6) (coe v13)) v7
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.ee-rm-/
d_ee'45'rm'45''47'_374 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] -> T_EE_20 -> MAlonzo.Code.Lang.T__'8712'__58 -> T_EE_20
d_ee'45'rm'45''47'_374 v0 ~v1 ~v2 v3 v4
  = du_ee'45'rm'45''47'_374 v0 v3 v4
du_ee'45'rm'45''47'_374 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_EE_20 -> MAlonzo.Code.Lang.T__'8712'__58 -> T_EE_20
du_ee'45'rm'45''47'_374 v0 v1 v2
  = case coe v1 of
      C_env_22 v5
        -> coe
             C_env_22 (coe du_env'45'rm'45''47'_356 (coe v0) (coe v5) (coe v2))
      C_let'8242'_24 v4 v6 v7
        -> coe
             C_let'8242'_24 v4 v6
             (coe du_ee'45'rm'45''47'_374 (coe v0) (coe v7) (coe v2))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.glet-sub
d_glet'45'sub_388 ::
  [Integer] ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8712'__58 -> MAlonzo.Code.Lang.T_Sub_522
d_glet'45'sub_388 v0 v1 v2
  = case coe v2 of
      MAlonzo.Code.Lang.C_here_60
        -> case coe v1 of
             MAlonzo.Code.Lang.C__'9657'__40 v5 v6
               -> coe
                    MAlonzo.Code.Lang.d_sub'45'id_560
                    (coe
                       MAlonzo.Code.Lang.C__'9657'__40
                       (coe
                          MAlonzo.Code.Lang.du__'47'__102
                          (coe
                             MAlonzo.Code.Lang.C__'9657'__40 (coe v5)
                             (coe MAlonzo.Code.Lang.C_ar_34 (coe v0)))
                          (coe MAlonzo.Code.Lang.C_here_60))
                       (coe MAlonzo.Code.Lang.C_ar_34 (coe v0)))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_there_62 v6
        -> case coe v1 of
             MAlonzo.Code.Lang.C__'9657'__40 v7 v8
               -> coe
                    MAlonzo.Code.Lang.d__'8729''738'__668
                    (coe
                       MAlonzo.Code.Lang.C__'9657'__40
                       (coe
                          MAlonzo.Code.Lang.C__'9657'__40
                          (coe MAlonzo.Code.Lang.du__'47'__102 (coe v7) (coe v6))
                          (coe MAlonzo.Code.Lang.C_ar_34 (coe v0)))
                       (coe v8))
                    (coe v1)
                    (coe
                       MAlonzo.Code.Lang.C__'9657'__40
                       (coe
                          MAlonzo.Code.Lang.du__'47'__102 (coe v1)
                          (coe MAlonzo.Code.Lang.C_there_62 v6))
                       (coe MAlonzo.Code.Lang.C_ar_34 (coe v0)))
                    (coe
                       MAlonzo.Code.Lang.d_skeep_544
                       (coe
                          MAlonzo.Code.Lang.C__'9657'__40
                          (coe MAlonzo.Code.Lang.du__'47'__102 (coe v7) (coe v6))
                          (coe MAlonzo.Code.Lang.C_ar_34 (coe v0)))
                       (coe v7) (coe v8)
                       (coe d_glet'45'sub_388 (coe v0) (coe v7) (coe v6)))
                    (coe
                       MAlonzo.Code.Lang.d_sub'45'swap_846
                       (coe MAlonzo.Code.Lang.du__'47'__102 (coe v7) (coe v6)) (coe v8)
                       (coe MAlonzo.Code.Lang.C_ar_34 (coe v0)))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.glet
d_glet_396 ::
  [Integer] ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 -> MAlonzo.Code.Lang.T_E_214
d_glet_396 v0 v1 v2 v3 v4 v5
  = let v6 = coe MAlonzo.Code.Lang.C_let'8242'_246 v0 v4 in
    coe
      (coe
         v6
         (MAlonzo.Code.Lang.d_sub_566
            (coe v1) (coe MAlonzo.Code.Lang.C_ar_34 (coe v2))
            (coe
               MAlonzo.Code.Lang.C__'9657'__40
               (coe MAlonzo.Code.Lang.du__'47'__102 (coe v1) (coe v3))
               (coe MAlonzo.Code.Lang.C_ar_34 (coe v0)))
            (coe v5) (coe d_glet'45'sub_388 (coe v0) (coe v1) (coe v3))))
-- Grad._.env-sub
d_env'45'sub_404 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_Env_12 -> MAlonzo.Code.Lang.T_Sub_522 -> T_Env_12
d_env'45'sub_404 v0 v1 v2 v3 v4
  = case coe v3 of
      C_ε_14 -> coe C_ε_14
      C_skip_16 v8
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v9 v10
               -> coe
                    C_skip_16
                    (d_env'45'sub_404 (coe v9) (coe v1) (coe v2) (coe v8) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      C__'9657'__18 v8 v9
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v10 v11
               -> coe
                    C__'9657'__18
                    (d_env'45'sub_404 (coe v10) (coe v1) (coe v2) (coe v8) (coe v4))
                    (MAlonzo.Code.Lang.d_sub_566
                       (coe v1) (coe v11) (coe v2) (coe v9) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.ee-sub
d_ee'45'sub_418 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  T_EE_20 -> MAlonzo.Code.Lang.T_Sub_522 -> T_EE_20
d_ee'45'sub_418 v0 v1 v2 v3 v4
  = case coe v3 of
      C_env_22 v7
        -> coe
             C_env_22
             (d_env'45'sub_404 (coe v0) (coe v1) (coe v2) (coe v7) (coe v4))
      C_let'8242'_24 v6 v8 v9
        -> coe
             C_let'8242'_24 v6
             (MAlonzo.Code.Lang.d_sub_566
                (coe v1) (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) (coe v2) (coe v8)
                (coe v4))
             (d_ee'45'sub_418
                (coe v0)
                (coe
                   MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)))
                (coe
                   MAlonzo.Code.Lang.C__'9657'__40 (coe v2)
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)))
                (coe v9)
                (coe
                   MAlonzo.Code.Lang.d_skeep_544 (coe v2) (coe v1)
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) (coe v4)))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.env-let
d_env'45'let_434 ::
  [Integer] ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  MAlonzo.Code.Lang.T_E_214 -> T_Env_12 -> T_EE_20
d_env'45'let_434 v0 v1 v2 v3 v4 v5
  = let v6 = coe C_let'8242'_24 v0 v4 in
    coe
      (coe
         v6
         (let v7 = coe C_env_22 in
          coe
            (coe
               v7
               (d_env'45'sub_404
                  (coe v2) (coe v1)
                  (coe
                     MAlonzo.Code.Lang.C__'9657'__40
                     (coe MAlonzo.Code.Lang.du__'47'__102 (coe v1) (coe v3))
                     (coe MAlonzo.Code.Lang.C_ar_34 (coe v0)))
                  (coe v5) (coe d_glet'45'sub_388 (coe v0) (coe v1) (coe v3))))))
-- Grad._.ee-let
d_ee'45'let_446 ::
  [Integer] ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  MAlonzo.Code.Lang.T_E_214 -> T_EE_20 -> T_EE_20
d_ee'45'let_446 v0 v1 v2 v3 v4 v5
  = coe
      C_let'8242'_24 v0 v4
      (d_ee'45'sub_418
         (coe v2) (coe v1)
         (coe
            MAlonzo.Code.Lang.C__'9657'__40
            (coe MAlonzo.Code.Lang.du__'47'__102 (coe v1) (coe v3))
            (coe MAlonzo.Code.Lang.C_ar_34 (coe v0)))
         (coe v5) (coe d_glet'45'sub_388 (coe v0) (coe v1) (coe v3)))
-- Grad._.ee-map-sum
d_ee'45'map'45'sum_454 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> [Integer] -> T_EE_20 -> T_EE_20
d_ee'45'map'45'sum_454 v0 v1 v2 v3
  = case coe v3 of
      C_env_22 v6
        -> coe
             C_env_22 (coe du_env'45'map'45'sum_166 (coe v0) (coe v2) (coe v6))
      C_let'8242'_24 v5 v7 v8
        -> let v9
                 = MAlonzo.Code.Lang.d_stren'45''8707'_876
                     (coe
                        MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                        (coe MAlonzo.Code.Lang.C_ix_32 (coe v2)))
                     (coe MAlonzo.Code.Lang.C_ar_34 (coe v5))
                     (coe MAlonzo.Code.Lang.C_ix_32 (coe v2)) (coe v7)
                     (coe MAlonzo.Code.Lang.C_here_60) in
           coe
             (case coe v9 of
                MAlonzo.Code.Agda.Builtin.Maybe.C_just_16 v10
                  -> case coe v10 of
                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v11 v12
                         -> coe
                              C_let'8242'_24 v5 v11
                              (d_ee'45'map'45'sum_454
                                 (coe v0)
                                 (coe
                                    MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                    (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)))
                                 (coe v2)
                                 (coe
                                    d_ee'45'sub_418 (coe v0)
                                    (coe
                                       MAlonzo.Code.Lang.C__'9657'__40
                                       (coe
                                          MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                          (coe MAlonzo.Code.Lang.C_ix_32 (coe v2)))
                                       (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)))
                                    (coe
                                       MAlonzo.Code.Lang.C__'9657'__40
                                       (coe
                                          MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                          (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)))
                                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v2)))
                                    (coe v8)
                                    (coe
                                       MAlonzo.Code.Lang.d_sub'45'swap_846 (coe v1)
                                       (coe MAlonzo.Code.Lang.C_ar_34 (coe v5))
                                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v2)))))
                       _ -> MAlonzo.RTE.mazUnreachableError
                MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                  -> coe
                       C_let'8242'_24 (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v2 v5)
                       (coe MAlonzo.Code.Lang.C_imap_226 v2 v5 v7)
                       (d_ee'45'map'45'sum_454
                          (coe v0)
                          (coe
                             MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                             (coe
                                MAlonzo.Code.Lang.C_ar_34
                                (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v2 v5)))
                          (coe v2)
                          (coe
                             d_ee'45'sub_418 (coe v0)
                             (coe
                                MAlonzo.Code.Lang.C__'9657'__40
                                (coe
                                   MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                   (coe MAlonzo.Code.Lang.C_ix_32 (coe v2)))
                                (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)))
                             (coe
                                MAlonzo.Code.Lang.C__'9657'__40
                                (coe
                                   MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                   (coe
                                      MAlonzo.Code.Lang.C_ar_34
                                      (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v2 v5)))
                                (coe MAlonzo.Code.Lang.C_ix_32 (coe v2)))
                             (coe v8)
                             (coe
                                MAlonzo.Code.Lang.C__'9657'__528
                                (MAlonzo.Code.Lang.d_skeep_544
                                   (coe
                                      MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                      (coe
                                         MAlonzo.Code.Lang.C_ar_34
                                         (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v2 v5)))
                                   (coe v1) (coe MAlonzo.Code.Lang.C_ix_32 (coe v2))
                                   (coe
                                      MAlonzo.Code.Lang.d_sdrop_540 (coe v1) (coe v1)
                                      (coe
                                         MAlonzo.Code.Lang.C_ar_34
                                         (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v2 v5))
                                      (coe MAlonzo.Code.Lang.d_sub'45'id_560 (coe v1))))
                                (coe
                                   MAlonzo.Code.Lang.C_sel_228 v2
                                   (coe
                                      MAlonzo.Code.Lang.C_var_216
                                      (coe
                                         MAlonzo.Code.Lang.C_there_62
                                         (coe MAlonzo.Code.Lang.C_here_60)))
                                   (coe
                                      MAlonzo.Code.Lang.C_var_216
                                      (coe MAlonzo.Code.Lang.C_here_60))))))
                _ -> MAlonzo.RTE.mazUnreachableError)
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.grad-last
d_grad'45'last_490 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] -> MAlonzo.Code.Lang.T_E_214 -> T_EE_20 -> T_EE_20
d_grad'45'last_490 v0 v1 v2 v3
  = case coe v3 of
      C_env_22 v6
        -> case coe v6 of
             C__'9657'__18 v10 v11
               -> let v12 = coe du_ee'45'tail_52 in
                  coe
                    (coe
                       v12
                       (let v13 = coe C_let'8242'_24 v1 v11 in
                        coe
                          (coe
                             v13
                             (coe
                                d_grad_500
                                (coe
                                   MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v1)))
                                (coe MAlonzo.Code.Lang.C_ar_34 (coe v1))
                                (coe
                                   MAlonzo.Code.Lang.d__'8593'_512 v0
                                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v1))
                                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v1)) v2)
                                (coe MAlonzo.Code.Lang.C_var_216 (coe MAlonzo.Code.Lang.C_here_60))
                                (let v14 = coe du_ee'45'push'45'zero_104 (coe v0) (coe v1) in
                                 coe
                                   (coe
                                      v14
                                      (d_ee'45'wk_40
                                         (coe v0)
                                         (coe
                                            MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v1)))
                                         (coe v0)
                                         (coe
                                            MAlonzo.Code.Lang.C_skip_356
                                            (MAlonzo.Code.Lang.d_'8838''45'eq_494 (coe v0)))
                                         (coe C_env_22 v10))))))))
             _ -> MAlonzo.RTE.mazUnreachableError
      C_let'8242'_24 v5 v7 v8
        -> let v9 = coe C_let'8242'_24 v5 v7 in
           coe
             (coe
                v9
                (let v10 = coe du_ee'45'tail_52 in
                 coe
                   (coe
                      v10
                      (d_grad'45'last_490
                         (coe
                            MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)))
                         (coe v1)
                         (coe
                            MAlonzo.Code.Lang.d__'8593'_512 v0
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v1))
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)) v2)
                         (coe
                            du_ee'45'wk'45'zero_92
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                               (coe MAlonzo.Code.Lang.C_ar_34 (coe v1)))
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__40
                               (coe
                                  MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                  (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)))
                               (coe MAlonzo.Code.Lang.C_ar_34 (coe v1)))
                            (coe v8)
                            (coe
                               MAlonzo.Code.Lang.C_keep_358
                               (coe
                                  MAlonzo.Code.Lang.C_skip_356
                                  (MAlonzo.Code.Lang.d_'8838''45'eq_494 (coe v0)))))))))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.grad-last′
d_grad'45'last'8242'_494 ::
  [Integer] ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  MAlonzo.Code.Lang.T_E_214 -> T_EE_20 -> T_EE_20
d_grad'45'last'8242'_494 v0 v1 v2 v3 v4
  = case coe v4 of
      C_env_22 v7
        -> coe
             du_ee'45'rm'45''47'_374 (coe v1)
             (coe
                d_ee'45'let_446 (coe v0) (coe v1) (coe v1) (coe v2)
                (coe du_env'45'lookup_338 (coe v1) (coe v7) (coe v2))
                (coe
                   d_grad_500 v1 (coe MAlonzo.Code.Lang.C_ar_34 (coe v0))
                   (MAlonzo.Code.Lang.d_wk_372
                      (coe MAlonzo.Code.Lang.du__'47'__102 (coe v1) (coe v2)) (coe v1)
                      (coe MAlonzo.Code.Lang.C_ar_34 (coe v0))
                      (coe
                         MAlonzo.Code.Lang.d_wk'45''47'_516
                         (coe MAlonzo.Code.Lang.C_ar_34 (coe v0)) (coe v1) (coe v2))
                      (coe v3))
                   (coe MAlonzo.Code.Lang.C_var_216 v2)
                   (coe
                      C_env_22
                      (d_env'45'wk_26
                         (coe MAlonzo.Code.Lang.du__'47'__102 (coe v1) (coe v2)) (coe v1)
                         (coe v1)
                         (coe
                            MAlonzo.Code.Lang.d_wk'45''47'_516
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v0)) (coe v1) (coe v2))
                         (coe v7)))))
             (coe v2)
      C_let'8242'_24 v6 v8 v9
        -> let v10 = coe C_let'8242'_24 v6 v8 in
           coe
             (coe
                v10
                (let v11 = coe du_ee'45'tail_52 in
                 coe
                   (coe
                      v11
                      (d_grad'45'last'8242'_494
                         (coe v0)
                         (coe
                            MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)))
                         (coe MAlonzo.Code.Lang.C_there_62 v2)
                         (coe
                            MAlonzo.Code.Lang.d__'8593'_512
                            (coe MAlonzo.Code.Lang.du__'47'__102 (coe v1) (coe v2))
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v0))
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) v3)
                         (coe du_ee'45'push'45'zero_104 (coe v1) (coe v6) (coe v9))))))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.grad
d_grad_500 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 -> T_EE_20 -> T_EE_20
d_grad_500 v0 v1 v2 v3
  = case coe v2 of
      MAlonzo.Code.Lang.C_var_216 v6
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ix_32 v7 -> coe (\ v8 -> v8)
             MAlonzo.Code.Lang.C_ar_34 v7
               -> coe
                    (\ v8 ->
                       d_ee'45'update'43'_150
                         (coe v0) (coe v0) (coe v7) (coe v8) (coe v6) (coe v3))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_zero_218 -> coe (\ v6 -> v6)
      MAlonzo.Code.Lang.C_one_220 -> coe (\ v6 -> v6)
      MAlonzo.Code.Lang.C_imaps_222 v6
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v7
               -> coe
                    (\ v8 ->
                       d_grad'45'sum_506
                         (coe v0) (coe v7) (coe MAlonzo.Code.Lang.d_unit_212) (coe v6)
                         (coe
                            MAlonzo.Code.Lang.C_sels_224 v7
                            (coe
                               MAlonzo.Code.Lang.d__'8593'_512 v0 v1
                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v7)) v3)
                            (coe
                               MAlonzo.Code.Lang.C_var_216 (coe MAlonzo.Code.Lang.C_here_60)))
                         (coe v8))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_sels_224 v5 v6 v7
        -> coe
             (\ v8 ->
                coe
                  d_grad_500 v0 (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)) v6
                  (coe
                     MAlonzo.Code.Lang.C_imaps_222
                     (coe
                        MAlonzo.Code.Lang.C_zero'45'but_236 v5
                        (coe MAlonzo.Code.Lang.C_var_216 (coe MAlonzo.Code.Lang.C_here_60))
                        (coe
                           MAlonzo.Code.Lang.d__'8593'_512 v0
                           (coe MAlonzo.Code.Lang.C_ix_32 (coe v5))
                           (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) v7)
                        (coe
                           MAlonzo.Code.Lang.d__'8593'_512 v0
                           (coe MAlonzo.Code.Lang.C_ar_34 (coe MAlonzo.Code.Lang.d_unit_212))
                           (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) v3)))
                  v8)
      MAlonzo.Code.Lang.C_imap_226 v5 v6 v7
        -> coe
             (\ v8 ->
                d_grad'45'sum_506
                  (coe v0) (coe v5) (coe v6) (coe v7)
                  (coe
                     MAlonzo.Code.Lang.C_sel_228 v5
                     (coe
                        MAlonzo.Code.Lang.d__'8593'_512 v0
                        (coe
                           MAlonzo.Code.Lang.C_ar_34
                           (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v5 v6))
                        (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) v3)
                     (coe
                        MAlonzo.Code.Lang.C_var_216 (coe MAlonzo.Code.Lang.C_here_60)))
                  (coe v8))
      MAlonzo.Code.Lang.C_sel_228 v5 v7 v8
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v9
               -> coe
                    (\ v10 ->
                       coe
                         d_grad_500 v0
                         (coe
                            MAlonzo.Code.Lang.C_ar_34
                            (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v5 v9))
                         v7
                         (coe
                            MAlonzo.Code.Lang.C_imap_226 v5 v9
                            (coe
                               MAlonzo.Code.Lang.C_zero'45'but_236 v5
                               (coe MAlonzo.Code.Lang.C_var_216 (coe MAlonzo.Code.Lang.C_here_60))
                               (coe
                                  MAlonzo.Code.Lang.d__'8593'_512 v0
                                  (coe MAlonzo.Code.Lang.C_ix_32 (coe v5))
                                  (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) v8)
                               (coe
                                  MAlonzo.Code.Lang.d__'8593'_512 v0 v1
                                  (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) v3)))
                         v10)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_imapb_230 v4 v5 v8 v9
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v10
               -> coe
                    (\ v11 ->
                       d_grad'45'sum_506
                         (coe v0) (coe v4) (coe v5) (coe v9)
                         (coe
                            MAlonzo.Code.Lang.C_selb_232 v4 v10 v8
                            (coe
                               MAlonzo.Code.Lang.d__'8593'_512 v0 v1
                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)) v3)
                            (coe
                               MAlonzo.Code.Lang.C_var_216 (coe MAlonzo.Code.Lang.C_here_60)))
                         (coe v11))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_selb_232 v4 v6 v8 v9 v10
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v11
               -> coe
                    (\ v12 ->
                       coe
                         d_grad_500 v0 (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) v9
                         (coe
                            MAlonzo.Code.Lang.C_imapb_230 v4 v11 v8
                            (coe
                               MAlonzo.Code.Lang.C_zero'45'but_236 v4
                               (coe MAlonzo.Code.Lang.C_var_216 (coe MAlonzo.Code.Lang.C_here_60))
                               (coe
                                  MAlonzo.Code.Lang.d__'8593'_512 v0
                                  (coe MAlonzo.Code.Lang.C_ix_32 (coe v4))
                                  (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)) v10)
                               (coe
                                  MAlonzo.Code.Lang.d__'8593'_512 v0 v1
                                  (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)) v3)))
                         v12)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_sum_234 v5 v7
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v8
               -> coe
                    (\ v9 ->
                       d_grad'45'sum_506
                         (coe v0) (coe v5) (coe v8) (coe v7)
                         (coe
                            MAlonzo.Code.Lang.d__'8593'_512 v0 v1
                            (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) v3)
                         (coe v9))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_zero'45'but_236 v5 v7 v8 v9
        -> coe
             (\ v10 ->
                coe
                  d_grad_500 v0 v1 v9
                  (coe MAlonzo.Code.Lang.C_zero'45'but_236 v5 v7 v8 v3) v10)
      MAlonzo.Code.Lang.C_slide_238 v5 v6 v7 v9 v10 v11 v12
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v13
               -> coe
                    (\ v14 ->
                       coe
                         d_grad_500 v0 (coe MAlonzo.Code.Lang.C_ar_34 (coe v7)) v11
                         (coe MAlonzo.Code.Lang.C_backslide_240 v5 v13 v6 v9 v3 v12 v10)
                         v14)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_backslide_240 v5 v6 v7 v9 v10 v11 v12
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v13
               -> coe
                    (\ v14 ->
                       coe
                         d_grad_500 v0 (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) v10
                         (coe MAlonzo.Code.Lang.C_slide_238 v5 v7 v13 v9 v12 v3 v11) v14)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_bin_242 v6 v7 v8
        -> case coe v6 of
             MAlonzo.Code.Lang.C_plus_190
               -> coe
                    (\ v9 ->
                       coe d_grad_500 v0 v1 v7 v3 (coe d_grad_500 v0 v1 v8 v3 v9))
             MAlonzo.Code.Lang.C_mul_192
               -> coe
                    (\ v9 ->
                       coe
                         d_grad_500 v0 v1 v7 (coe MAlonzo.Code.Lang.C_bin_242 v6 v3 v8)
                         (coe
                            d_grad_500 v0 v1 v8 (coe MAlonzo.Code.Lang.C_bin_242 v6 v7 v3) v9))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_scaledown_244 v6 v7
        -> coe
             d_grad_500 (coe v0) (coe v1) (coe v7)
             (coe MAlonzo.Code.Lang.C_scaledown_244 v6 v3)
      MAlonzo.Code.Lang.C_let'8242'_246 v5 v7 v8
        -> coe
             (\ v9 ->
                d_grad'45'last_490
                  (coe v0) (coe v5) (coe v7)
                  (coe
                     C_let'8242'_24 v5 v7
                     (coe
                        d_grad_500
                        (coe
                           MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                           (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)))
                        v1 v8
                        (coe
                           MAlonzo.Code.Lang.d__'8593'_512 v0 v1
                           (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)) v3)
                        (let v10 = coe du_ee'45'push'45'zero_104 (coe v0) (coe v5) in
                         coe
                           (coe
                              v10
                              (d_ee'45'wk_40
                                 (coe v0)
                                 (coe
                                    MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                    (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)))
                                 (coe v0)
                                 (coe
                                    MAlonzo.Code.Lang.C_skip_356
                                    (MAlonzo.Code.Lang.d_'8838''45'eq_494 (coe v0)))
                                 (coe v9)))))))
      MAlonzo.Code.Lang.C_un_248 v6 v7
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v8
               -> case coe v6 of
                    MAlonzo.Code.Lang.C_logistic_196
                      -> coe
                           d_grad_500 (coe v0) (coe v1) (coe v7)
                           (coe
                              MAlonzo.Code.Lang.C_let'8242'_246 v8
                              (coe MAlonzo.Code.Lang.C_un_248 v6 v7)
                              (coe
                                 MAlonzo.Code.Lang.C_bin_242 (coe MAlonzo.Code.Lang.C_mul_192)
                                 (coe
                                    MAlonzo.Code.Lang.C_bin_242 (coe MAlonzo.Code.Lang.C_mul_192)
                                    (coe MAlonzo.Code.Lang.d__'8593'_512 v0 v1 v1 v3)
                                    (coe
                                       MAlonzo.Code.Lang.C_var_216
                                       (coe MAlonzo.Code.Lang.C_here_60)))
                                 (coe
                                    MAlonzo.Code.Lang.C_bin_242 (coe MAlonzo.Code.Lang.C_plus_190)
                                    (coe MAlonzo.Code.Lang.C_one_220)
                                    (coe
                                       MAlonzo.Code.Lang.C_un_248 (coe MAlonzo.Code.Lang.C_neg_198)
                                       (coe
                                          MAlonzo.Code.Lang.C_var_216
                                          (coe MAlonzo.Code.Lang.C_here_60))))))
                    MAlonzo.Code.Lang.C_neg_198
                      -> coe
                           d_grad_500 (coe v0) (coe v1) (coe v7)
                           (coe MAlonzo.Code.Lang.C_un_248 v6 v3)
                    MAlonzo.Code.Lang.C_rectifier_200
                      -> coe
                           (\ v9 ->
                              coe
                                d_grad_500 v0 v1 v7
                                (coe
                                   MAlonzo.Code.Lang.C_bin_242 (coe MAlonzo.Code.Lang.C_mul_192)
                                   (coe
                                      MAlonzo.Code.Lang.C_un_248
                                      (coe MAlonzo.Code.Lang.C_ind'45'positive_206) v7)
                                   v3)
                                v9)
                    MAlonzo.Code.Lang.C_squared_202
                      -> coe
                           (\ v9 ->
                              coe
                                d_grad_500 v0 v1 v7
                                (coe
                                   MAlonzo.Code.Lang.du__'47''47'__322 (coe v3)
                                   (coe
                                      MAlonzo.Code.Lang.C_bin_242 (coe MAlonzo.Code.Lang.C_mul_192)
                                      (coe MAlonzo.Code.Lang.du_𝟚_342)
                                      (coe MAlonzo.Code.Lang.C_un_248 v6 v7)))
                                v9)
                    MAlonzo.Code.Lang.C_inverse_204
                      -> coe
                           (\ v9 ->
                              coe
                                d_grad_500 v0 v1 v7
                                (coe
                                   MAlonzo.Code.Lang.C_un_248 (coe MAlonzo.Code.Lang.C_neg_198)
                                   (coe
                                      MAlonzo.Code.Lang.C_bin_242 (coe MAlonzo.Code.Lang.C_mul_192)
                                      (coe MAlonzo.Code.Lang.C_un_248 v6 v7)
                                      (coe MAlonzo.Code.Lang.du__'47''47'__322 (coe v3) (coe v7))))
                                v9)
                    MAlonzo.Code.Lang.C_ind'45'positive_206
                      -> coe
                           (\ v9 ->
                              coe d_grad_500 v0 v1 v7 (coe MAlonzo.Code.Lang.C_zero_218) v9)
                    MAlonzo.Code.Lang.C_logarithm_208
                      -> coe
                           (\ v9 ->
                              coe
                                d_grad_500 v0 v1 v7
                                (coe MAlonzo.Code.Lang.du__'47''47'__322 (coe v3) (coe v7)) v9)
                    MAlonzo.Code.Lang.C_softmax_210
                      -> coe
                           (\ v9 ->
                              coe
                                du_ee'45'tail_52
                                (coe
                                   du_ee'45'tail_52
                                   (coe
                                      du_ee'45'tail_52
                                      (coe
                                         C_let'8242'_24 v8 (coe MAlonzo.Code.Lang.C_un_248 v6 v7)
                                         (coe
                                            C_let'8242'_24 v8
                                            (coe MAlonzo.Code.Lang.d__'8593'_512 v0 v1 v1 v3)
                                            (coe
                                               C_let'8242'_24 MAlonzo.Code.Lang.d_unit_212
                                               (coe
                                                  MAlonzo.Code.Lang.C_sum_234 v8
                                                  (coe
                                                     MAlonzo.Code.Lang.C_sels_224 v8
                                                     (coe
                                                        MAlonzo.Code.Lang.C_bin_242
                                                        (coe MAlonzo.Code.Lang.C_mul_192)
                                                        (coe
                                                           MAlonzo.Code.Lang.C_var_216
                                                           (coe
                                                              MAlonzo.Code.Lang.C_there_62
                                                              (coe MAlonzo.Code.Lang.C_here_60)))
                                                        (coe
                                                           MAlonzo.Code.Lang.C_var_216
                                                           (coe
                                                              MAlonzo.Code.Lang.C_there_62
                                                              (coe
                                                                 MAlonzo.Code.Lang.C_there_62
                                                                 (coe
                                                                    MAlonzo.Code.Lang.C_here_60)))))
                                                     (coe
                                                        MAlonzo.Code.Lang.C_var_216
                                                        (coe MAlonzo.Code.Lang.C_here_60))))
                                               (coe
                                                  d_grad_500
                                                  (coe
                                                     MAlonzo.Code.Lang.C__'9657'__40
                                                     (coe
                                                        MAlonzo.Code.Lang.C__'9657'__40
                                                        (coe
                                                           MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                                           (coe v1))
                                                        (coe v1))
                                                     (coe
                                                        MAlonzo.Code.Lang.C_ar_34
                                                        (coe MAlonzo.Code.Lang.d_unit_212)))
                                                  v1
                                                  (MAlonzo.Code.Lang.d_wk_372
                                                     (coe v0)
                                                     (coe
                                                        MAlonzo.Code.Lang.C__'9657'__40
                                                        (coe
                                                           MAlonzo.Code.Lang.C__'9657'__40
                                                           (coe
                                                              MAlonzo.Code.Lang.C__'9657'__40
                                                              (coe v0) (coe v1))
                                                           (coe v1))
                                                        (coe
                                                           MAlonzo.Code.Lang.C_ar_34
                                                           (coe MAlonzo.Code.Lang.d_unit_212)))
                                                     (coe v1)
                                                     (coe
                                                        MAlonzo.Code.Lang.C_skip_356
                                                        (coe
                                                           MAlonzo.Code.Lang.C_skip_356
                                                           (coe
                                                              MAlonzo.Code.Lang.C_skip_356
                                                              (MAlonzo.Code.Lang.d_'8838''45'eq_494
                                                                 (coe v0)))))
                                                     (coe v7))
                                                  (coe
                                                     MAlonzo.Code.Lang.C_bin_242
                                                     (coe MAlonzo.Code.Lang.C_mul_192)
                                                     (coe
                                                        MAlonzo.Code.Lang.C_var_216
                                                        (coe
                                                           MAlonzo.Code.Lang.C_there_62
                                                           (coe
                                                              MAlonzo.Code.Lang.C_there_62
                                                              (coe MAlonzo.Code.Lang.C_here_60))))
                                                     (coe
                                                        MAlonzo.Code.Lang.du__'8863'__302
                                                        (coe
                                                           MAlonzo.Code.Lang.C_var_216
                                                           (coe
                                                              MAlonzo.Code.Lang.C_there_62
                                                              (coe MAlonzo.Code.Lang.C_here_60)))
                                                        (coe
                                                           MAlonzo.Code.Lang.C_imaps_222
                                                           (coe
                                                              MAlonzo.Code.Lang.C_var_216
                                                              (coe
                                                                 MAlonzo.Code.Lang.C_there_62
                                                                 (coe
                                                                    MAlonzo.Code.Lang.C_here_60))))))
                                                  (coe
                                                     du_ee'45'wk'45'zero_92 (coe v0)
                                                     (coe
                                                        MAlonzo.Code.Lang.C__'9657'__40
                                                        (coe
                                                           MAlonzo.Code.Lang.C__'9657'__40
                                                           (coe
                                                              MAlonzo.Code.Lang.C__'9657'__40
                                                              (coe v0) (coe v1))
                                                           (coe v1))
                                                        (coe
                                                           MAlonzo.Code.Lang.C_ar_34
                                                           (coe MAlonzo.Code.Lang.d_unit_212)))
                                                     (coe
                                                        d_ee'45'wk_40 (coe v0)
                                                        (coe
                                                           MAlonzo.Code.Lang.C__'9657'__40
                                                           (coe
                                                              MAlonzo.Code.Lang.C__'9657'__40
                                                              (coe
                                                                 MAlonzo.Code.Lang.C__'9657'__40
                                                                 (coe v0) (coe v1))
                                                              (coe v1))
                                                           (coe
                                                              MAlonzo.Code.Lang.C_ar_34
                                                              (coe MAlonzo.Code.Lang.d_unit_212)))
                                                        (coe v0)
                                                        (coe
                                                           MAlonzo.Code.Lang.C_skip_356
                                                           (coe
                                                              MAlonzo.Code.Lang.C_skip_356
                                                              (coe
                                                                 MAlonzo.Code.Lang.C_skip_356
                                                                 (MAlonzo.Code.Lang.d_'8838''45'eq_494
                                                                    (coe v0)))))
                                                        (coe v9))
                                                     (coe
                                                        MAlonzo.Code.Lang.C_skip_356
                                                        (coe
                                                           MAlonzo.Code.Lang.C_skip_356
                                                           (coe
                                                              MAlonzo.Code.Lang.C_skip_356
                                                              (MAlonzo.Code.Lang.d_'8838''45'eq_494
                                                                 (coe v0)))))))))))))
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Grad._.grad-sum
d_grad'45'sum_506 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 -> T_EE_20 -> T_EE_20
d_grad'45'sum_506 v0 v1 v2 v3 v4 v5
  = coe
      d_ee'45'plus_238 (coe v0) (coe v0) (coe v5)
      (coe
         du_ee'45'tail_52
         (coe
            d_ee'45'map'45'sum_454
            (coe
               MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
               (coe MAlonzo.Code.Lang.C_ix_32 (coe v1)))
            (coe v0) (coe v1)
            (coe
               d_grad_500
               (coe
                  MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                  (coe MAlonzo.Code.Lang.C_ix_32 (coe v1)))
               (coe MAlonzo.Code.Lang.C_ar_34 (coe v2)) v3 v4
               (coe
                  du_zero'45'ee_118
                  (coe
                     MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                     (coe MAlonzo.Code.Lang.C_ix_32 (coe v1)))))))
