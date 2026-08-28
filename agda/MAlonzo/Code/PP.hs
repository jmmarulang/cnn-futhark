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

module MAlonzo.Code.PP where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.Sigma
import qualified MAlonzo.Code.Agda.Builtin.String
import qualified MAlonzo.Code.Agda.Primitive
import qualified MAlonzo.Code.Ar
import qualified MAlonzo.Code.Data.Nat.Properties
import qualified MAlonzo.Code.Data.Nat.Show
import qualified MAlonzo.Code.Data.String.Base
import qualified MAlonzo.Code.Effect.Applicative
import qualified MAlonzo.Code.Effect.Monad
import qualified MAlonzo.Code.Effect.Monad.Identity
import qualified MAlonzo.Code.Effect.Monad.State
import qualified MAlonzo.Code.Effect.Monad.State.Transformer
import qualified MAlonzo.Code.Effect.Monad.State.Transformer.Base
import qualified MAlonzo.Code.Function.Base
import qualified MAlonzo.Code.Lang
import qualified MAlonzo.Code.Relation.Nullary.Decidable.Core
import qualified MAlonzo.Code.Text.Printf

-- PP._
d___64 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> MAlonzo.Code.Effect.Monad.T_RawMonad_24
d___64 v0 v1 = coe MAlonzo.Code.Effect.Monad.State.du_monad_42
-- PP._
d___68 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28
d___68 v0 v1 = coe MAlonzo.Code.Effect.Monad.State.du_monadState_46
-- PP.Sem
d_Sem_70 :: MAlonzo.Code.Lang.T_IS_30 -> ()
d_Sem_70 = erased
-- PP.FEnv
d_FEnv_74 :: MAlonzo.Code.Lang.T_Ctx_36 -> ()
d_FEnv_74 = erased
-- PP.lookup
d_lookup_80 ::
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.String.T_String_6
d_lookup_80 ~v0 v1 v2 v3 = du_lookup_80 v1 v2 v3
du_lookup_80 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.String.T_String_6
du_lookup_80 v0 v1 v2
  = case coe v1 of
      MAlonzo.Code.Lang.C_here_60
        -> case coe v2 of
             MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v5 v6 -> coe v6
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_there_62 v6
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v7 v8
               -> case coe v2 of
                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v9 v10
                      -> coe du_lookup_80 (coe v7) (coe v6) (coe v9)
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- PP.fresh-name
d_fresh'45'name_92 ::
  Integer -> MAlonzo.Code.Agda.Builtin.String.T_String_6
d_fresh'45'name_92 v0
  = coe
      MAlonzo.Code.Data.String.Base.d__'43''43'__20
      ("x" :: Data.Text.Text)
      (coe MAlonzo.Code.Data.Nat.Show.d_show_56 v0)
-- PP.fresh-var
d_fresh'45'var_96 ::
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58
d_fresh'45'var_96
  = coe
      MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
      (coe
         (\ v0 ->
            coe
              MAlonzo.Code.Function.Base.du__'8728''8242'__216
              (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
              (\ v1 ->
                 MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v1))
              (coe
                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                 (coe
                    MAlonzo.Code.Effect.Monad.State.Transformer.Base.du_get_44
                    (coe
                       MAlonzo.Code.Effect.Monad.State.Transformer.du_monadState_460
                       (coe MAlonzo.Code.Effect.Monad.Identity.du_monad_36)))
                 v0)
              (\ v1 ->
                 case coe v1 of
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v2 v3
                     -> coe
                          MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                          (coe
                             MAlonzo.Code.Effect.Monad.du__'62''62'__70 (coe d___64 () erased)
                             (coe
                                MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_modify_40
                                (coe d___68 () erased)
                                (\ v4 -> addInt (coe (1 :: Integer)) (coe v4)))
                             (coe
                                MAlonzo.Code.Effect.Applicative.du_return_68
                                (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                   (coe d___64 () erased))
                                (d_fresh'45'name_92 (coe v3))))
                          v2
                   _ -> MAlonzo.RTE.mazUnreachableError)))
-- PP.bop
d_bop_100 ::
  MAlonzo.Code.Lang.T_Bop_188 ->
  MAlonzo.Code.Agda.Builtin.String.T_String_6
d_bop_100 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_plus_190 -> coe ("+" :: Data.Text.Text)
      MAlonzo.Code.Lang.C_mul_192 -> coe ("*" :: Data.Text.Text)
      _ -> MAlonzo.RTE.mazUnreachableError
-- PP.uop
d_uop_102 ::
  MAlonzo.Code.Lang.T_Uop_194 ->
  MAlonzo.Code.Agda.Builtin.String.T_String_6
d_uop_102 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_logistic_196 -> coe ("log" :: Data.Text.Text)
      MAlonzo.Code.Lang.C_neg_198 -> coe ("-" :: Data.Text.Text)
      MAlonzo.Code.Lang.C_rectifier_200 -> coe ("relu" :: Data.Text.Text)
      MAlonzo.Code.Lang.C_squared_202 -> coe ("sqrt" :: Data.Text.Text)
      MAlonzo.Code.Lang.C_inverse_204 -> coe ("inv" :: Data.Text.Text)
      MAlonzo.Code.Lang.C_ind'45'positive_206
        -> coe ("ind-positive" :: Data.Text.Text)
      MAlonzo.Code.Lang.C_logarithm_208 -> coe ("ln" :: Data.Text.Text)
      MAlonzo.Code.Lang.C_softmax_210
        -> coe ("softmax" :: Data.Text.Text)
      _ -> MAlonzo.RTE.mazUnreachableError
-- PP.pars
d_pars_104 ::
  Bool ->
  MAlonzo.Code.Agda.Builtin.String.T_String_6 ->
  MAlonzo.Code.Agda.Builtin.String.T_String_6
d_pars_104 v0
  = if coe v0
      then coe
             MAlonzo.Code.Text.Printf.d_printf_26
             (coe ("(%s)" :: Data.Text.Text))
      else coe (\ v1 -> v1)
-- PP.precImap
d_precImap_106 :: Integer
d_precImap_106 = coe (1 :: Integer)
-- PP.precLet
d_precLet_108 :: Integer
d_precLet_108 = coe (2 :: Integer)
-- PP.precAdd
d_precAdd_110 :: Integer
d_precAdd_110 = coe (3 :: Integer)
-- PP.precMul
d_precMul_112 :: Integer
d_precMul_112 = coe (4 :: Integer)
-- PP.precApp
d_precApp_114 :: Integer
d_precApp_114 = coe (6 :: Integer)
-- PP.ppx
d_ppx_118 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  Integer ->
  MAlonzo.Code.Lang.T_E_214 ->
  AgdaAny ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58
d_ppx_118 v0 v1 v2 v3 v4
  = case coe v3 of
      MAlonzo.Code.Lang.C_var_216 v7
        -> coe
             MAlonzo.Code.Effect.Applicative.du_return_68
             (coe
                MAlonzo.Code.Effect.Monad.State.Transformer.du_applicative_48
                (coe MAlonzo.Code.Effect.Monad.Identity.du_monad_36))
             (coe du_lookup_80 (coe v0) (coe v7) (coe v4))
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Effect.Applicative.du_return_68
             (coe
                MAlonzo.Code.Effect.Monad.State.Transformer.du_applicative_48
                (coe MAlonzo.Code.Effect.Monad.Identity.du_monad_36))
             ("0" :: Data.Text.Text)
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Effect.Applicative.du_return_68
             (coe
                MAlonzo.Code.Effect.Monad.State.Transformer.du_applicative_48
                (coe MAlonzo.Code.Effect.Monad.Identity.du_monad_36))
             ("1" :: Data.Text.Text)
      MAlonzo.Code.Lang.C_imaps_222 v7
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v8
               -> coe
                    MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
                    (coe
                       (\ v9 ->
                          coe
                            MAlonzo.Code.Function.Base.du__'8728''8242'__216
                            (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                            (\ v10 ->
                               MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v10))
                            (coe
                               MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                               d_fresh'45'var_96 v9)
                            (\ v10 ->
                               case coe v10 of
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v11 v12
                                   -> coe
                                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                        (coe
                                           MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                           (coe d___64 () erased) erased erased
                                           (d_ppx_118
                                              (coe
                                                 MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                                 (coe MAlonzo.Code.Lang.C_ix_32 (coe v8)))
                                              (coe
                                                 MAlonzo.Code.Lang.C_ar_34
                                                 (coe MAlonzo.Code.Lang.d_unit_212))
                                              (coe (0 :: Integer)) (coe v7)
                                              (coe
                                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                 (coe v4) (coe v12)))
                                           (\ v13 ->
                                              coe
                                                MAlonzo.Code.Effect.Applicative.du_return_68
                                                (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                                   (coe d___64 () erased))
                                                (coe
                                                   d_pars_104
                                                   (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                                      (coe
                                                         MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                         (coe v2) (coe d_precImap_106)))
                                                   (coe
                                                      MAlonzo.Code.Text.Printf.d_printf_26
                                                      ("imaps \955 %s \8594 %s" :: Data.Text.Text)
                                                      v12 v13))))
                                        v11
                                 _ -> MAlonzo.RTE.mazUnreachableError)))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_sels_224 v6 v7 v8
        -> coe
             MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
             (coe
                (\ v9 ->
                   coe
                     MAlonzo.Code.Function.Base.du__'8728''8242'__216
                     (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                     (\ v10 ->
                        MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v10))
                     (coe
                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                        (d_ppx_118
                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v6))
                           (coe addInt (coe (1 :: Integer)) (coe d_precApp_114)) (coe v7)
                           (coe v4))
                        v9)
                     (\ v10 ->
                        case coe v10 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v11 v12
                            -> coe
                                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                 (coe
                                    MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                    (coe d___64 () erased) erased erased
                                    (d_ppx_118
                                       (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v6))
                                       (coe addInt (coe (1 :: Integer)) (coe d_precApp_114))
                                       (coe v8) (coe v4))
                                    (\ v13 ->
                                       coe
                                         MAlonzo.Code.Effect.Applicative.du_return_68
                                         (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                            (coe d___64 () erased))
                                         (coe
                                            d_pars_104
                                            (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                               (coe
                                                  MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                  (coe v2) (coe d_precApp_114)))
                                            (coe
                                               MAlonzo.Code.Text.Printf.d_printf_26
                                               ("sels %s %s" :: Data.Text.Text) v12 v13))))
                                 v11
                          _ -> MAlonzo.RTE.mazUnreachableError)))
      MAlonzo.Code.Lang.C_imap_226 v6 v7 v8
        -> coe
             MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
             (coe
                (\ v9 ->
                   coe
                     MAlonzo.Code.Function.Base.du__'8728''8242'__216
                     (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                     (\ v10 ->
                        MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v10))
                     (coe
                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                        d_fresh'45'var_96 v9)
                     (\ v10 ->
                        case coe v10 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v11 v12
                            -> coe
                                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                 (coe
                                    MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                    (coe d___64 () erased) erased erased
                                    (d_ppx_118
                                       (coe
                                          MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                          (coe MAlonzo.Code.Lang.C_ix_32 (coe v6)))
                                       (coe MAlonzo.Code.Lang.C_ar_34 (coe v7)) (coe (0 :: Integer))
                                       (coe v8)
                                       (coe
                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                                          (coe v12)))
                                    (\ v13 ->
                                       coe
                                         MAlonzo.Code.Effect.Applicative.du_return_68
                                         (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                            (coe d___64 () erased))
                                         (coe
                                            d_pars_104
                                            (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                               (coe
                                                  MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                  (coe v2) (coe d_precImap_106)))
                                            (coe
                                               MAlonzo.Code.Text.Printf.d_printf_26
                                               ("imap \955 %s \8594 %s" :: Data.Text.Text) v12
                                               v13))))
                                 v11
                          _ -> MAlonzo.RTE.mazUnreachableError)))
      MAlonzo.Code.Lang.C_sel_228 v6 v8 v9
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v10
               -> coe
                    MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
                    (coe
                       (\ v11 ->
                          coe
                            MAlonzo.Code.Function.Base.du__'8728''8242'__216
                            (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                            (\ v12 ->
                               MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v12))
                            (coe
                               MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                               (d_ppx_118
                                  (coe v0)
                                  (coe
                                     MAlonzo.Code.Lang.C_ar_34
                                     (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v6 v10))
                                  (coe addInt (coe (1 :: Integer)) (coe d_precApp_114)) (coe v8)
                                  (coe v4))
                               v11)
                            (\ v12 ->
                               case coe v12 of
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v13 v14
                                   -> coe
                                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                        (coe
                                           MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                           (coe d___64 () erased) erased erased
                                           (d_ppx_118
                                              (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v6))
                                              (coe addInt (coe (1 :: Integer)) (coe d_precApp_114))
                                              (coe v9) (coe v4))
                                           (\ v15 ->
                                              coe
                                                MAlonzo.Code.Effect.Applicative.du_return_68
                                                (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                                   (coe d___64 () erased))
                                                (coe
                                                   d_pars_104
                                                   (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                                      (coe
                                                         MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                         (coe v2) (coe d_precApp_114)))
                                                   (coe
                                                      MAlonzo.Code.Text.Printf.d_printf_26
                                                      ("sel %s %s" :: Data.Text.Text) v14 v15))))
                                        v13
                                 _ -> MAlonzo.RTE.mazUnreachableError)))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_imapb_230 v5 v6 v9 v10
        -> coe
             MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
             (coe
                (\ v11 ->
                   coe
                     MAlonzo.Code.Function.Base.du__'8728''8242'__216
                     (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                     (\ v12 ->
                        MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v12))
                     (coe
                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                        d_fresh'45'var_96 v11)
                     (\ v12 ->
                        case coe v12 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v13 v14
                            -> coe
                                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                 (coe
                                    MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                    (coe d___64 () erased) erased erased
                                    (d_ppx_118
                                       (coe
                                          MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                          (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                                       (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) (coe (0 :: Integer))
                                       (coe v10)
                                       (coe
                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                                          (coe v14)))
                                    (\ v15 ->
                                       coe
                                         MAlonzo.Code.Effect.Applicative.du_return_68
                                         (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                            (coe d___64 () erased))
                                         (coe
                                            d_pars_104
                                            (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                               (coe
                                                  MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                  (coe v2) (coe d_precImap_106)))
                                            (coe
                                               MAlonzo.Code.Text.Printf.d_printf_26
                                               ("imapb \955 %s \8594 %s" :: Data.Text.Text) v14
                                               v15))))
                                 v13
                          _ -> MAlonzo.RTE.mazUnreachableError)))
      MAlonzo.Code.Lang.C_selb_232 v5 v7 v9 v10 v11
        -> coe
             MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
             (coe
                (\ v12 ->
                   coe
                     MAlonzo.Code.Function.Base.du__'8728''8242'__216
                     (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                     (\ v13 ->
                        MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v13))
                     (coe
                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                        (d_ppx_118
                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v7))
                           (coe addInt (coe (1 :: Integer)) (coe d_precApp_114)) (coe v10)
                           (coe v4))
                        v12)
                     (\ v13 ->
                        case coe v13 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                            -> coe
                                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                 (coe
                                    MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                    (coe d___64 () erased) erased erased
                                    (d_ppx_118
                                       (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v5))
                                       (coe addInt (coe (1 :: Integer)) (coe d_precApp_114))
                                       (coe v11) (coe v4))
                                    (\ v16 ->
                                       coe
                                         MAlonzo.Code.Effect.Applicative.du_return_68
                                         (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                            (coe d___64 () erased))
                                         (coe
                                            d_pars_104
                                            (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                               (coe
                                                  MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                  (coe v2) (coe d_precApp_114)))
                                            (coe
                                               MAlonzo.Code.Text.Printf.d_printf_26
                                               ("selb %s %s" :: Data.Text.Text) v15 v16))))
                                 v14
                          _ -> MAlonzo.RTE.mazUnreachableError)))
      MAlonzo.Code.Lang.C_sum_234 v6 v8
        -> coe
             MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
             (coe
                (\ v9 ->
                   coe
                     MAlonzo.Code.Function.Base.du__'8728''8242'__216
                     (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                     (\ v10 ->
                        MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v10))
                     (coe
                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                        d_fresh'45'var_96 v9)
                     (\ v10 ->
                        case coe v10 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v11 v12
                            -> coe
                                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                 (coe
                                    MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                    (coe d___64 () erased) erased erased
                                    (d_ppx_118
                                       (coe
                                          MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                          (coe MAlonzo.Code.Lang.C_ix_32 (coe v6)))
                                       (coe v1) (coe (0 :: Integer)) (coe v8)
                                       (coe
                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                                          (coe v12)))
                                    (\ v13 ->
                                       coe
                                         MAlonzo.Code.Effect.Applicative.du_return_68
                                         (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                            (coe d___64 () erased))
                                         (coe
                                            d_pars_104
                                            (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                               (coe
                                                  MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                  (coe v2) (coe d_precImap_106)))
                                            (coe
                                               MAlonzo.Code.Text.Printf.d_printf_26
                                               ("sum \955 %s \8594 %s" :: Data.Text.Text) v12
                                               v13))))
                                 v11
                          _ -> MAlonzo.RTE.mazUnreachableError)))
      MAlonzo.Code.Lang.C_zero'45'but_236 v6 v8 v9 v10
        -> coe
             MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
             (coe
                (\ v11 ->
                   coe
                     MAlonzo.Code.Function.Base.du__'8728''8242'__216
                     (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                     (\ v12 ->
                        MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v12))
                     (coe
                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                        (d_ppx_118
                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v6))
                           (coe addInt (coe (1 :: Integer)) (coe d_precApp_114)) (coe v8)
                           (coe v4))
                        v11)
                     (\ v12 ->
                        case coe v12 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v13 v14
                            -> coe
                                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                 (coe
                                    MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                    (coe d___64 () erased) erased erased
                                    (d_ppx_118
                                       (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v6))
                                       (coe addInt (coe (1 :: Integer)) (coe d_precApp_114))
                                       (coe v9) (coe v4))
                                    (\ v15 ->
                                       coe
                                         MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                         (coe d___64 () erased) erased erased
                                         (d_ppx_118
                                            (coe v0) (coe v1)
                                            (coe addInt (coe (1 :: Integer)) (coe d_precApp_114))
                                            (coe v10) (coe v4))
                                         (\ v16 ->
                                            coe
                                              MAlonzo.Code.Effect.Applicative.du_return_68
                                              (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                                 (coe d___64 () erased))
                                              (coe
                                                 d_pars_104
                                                 (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                                    (coe
                                                       MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                       (coe v2) (coe d_precApp_114)))
                                                 (coe
                                                    MAlonzo.Code.Text.Printf.d_printf_26
                                                    ("(zero-but %s %s %s)" :: Data.Text.Text) v14
                                                    v15 v16)))))
                                 v13
                          _ -> MAlonzo.RTE.mazUnreachableError)))
      MAlonzo.Code.Lang.C_slide_238 v6 v7 v8 v10 v11 v12 v13
        -> coe
             MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
             (coe
                (\ v14 ->
                   coe
                     MAlonzo.Code.Function.Base.du__'8728''8242'__216
                     (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                     (\ v15 ->
                        MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v15))
                     (coe
                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                        (d_ppx_118
                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v6))
                           (coe addInt (coe (1 :: Integer)) (coe d_precApp_114)) (coe v10)
                           (coe v4))
                        v14)
                     (\ v15 ->
                        case coe v15 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                            -> coe
                                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                 (coe
                                    MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                    (coe d___64 () erased) erased erased
                                    (d_ppx_118
                                       (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v8))
                                       (coe addInt (coe (1 :: Integer)) (coe d_precApp_114))
                                       (coe v12) (coe v4))
                                    (\ v18 ->
                                       coe
                                         MAlonzo.Code.Effect.Applicative.du_return_68
                                         (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                            (coe d___64 () erased))
                                         (coe
                                            d_pars_104
                                            (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                               (coe
                                                  MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                  (coe v2) (coe d_precApp_114)))
                                            (coe
                                               MAlonzo.Code.Text.Printf.d_printf_26
                                               ("slide %s %s" :: Data.Text.Text) v17 v18))))
                                 v16
                          _ -> MAlonzo.RTE.mazUnreachableError)))
      MAlonzo.Code.Lang.C_backslide_240 v6 v7 v8 v10 v11 v12 v13
        -> coe
             MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
             (coe
                (\ v14 ->
                   coe
                     MAlonzo.Code.Function.Base.du__'8728''8242'__216
                     (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                     (\ v15 ->
                        MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v15))
                     (coe
                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                        (d_ppx_118
                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v6))
                           (coe addInt (coe (1 :: Integer)) (coe d_precApp_114)) (coe v10)
                           (coe v4))
                        v14)
                     (\ v15 ->
                        case coe v15 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                            -> coe
                                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                 (coe
                                    MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                    (coe d___64 () erased) erased erased
                                    (d_ppx_118
                                       (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v7))
                                       (coe addInt (coe (1 :: Integer)) (coe d_precApp_114))
                                       (coe v11) (coe v4))
                                    (\ v18 ->
                                       coe
                                         MAlonzo.Code.Effect.Applicative.du_return_68
                                         (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                            (coe d___64 () erased))
                                         (coe
                                            d_pars_104
                                            (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                               (coe
                                                  MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                  (coe v2) (coe d_precApp_114)))
                                            (coe
                                               MAlonzo.Code.Text.Printf.d_printf_26
                                               ("backslide %s %s" :: Data.Text.Text) v17 v18))))
                                 v16
                          _ -> MAlonzo.RTE.mazUnreachableError)))
      MAlonzo.Code.Lang.C_bin_242 v7 v8 v9
        -> case coe v7 of
             MAlonzo.Code.Lang.C_plus_190
               -> coe
                    MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
                    (coe
                       (\ v10 ->
                          coe
                            MAlonzo.Code.Function.Base.du__'8728''8242'__216
                            (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                            (\ v11 ->
                               MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v11))
                            (coe
                               MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                               (d_ppx_118 (coe v0) (coe v1) (coe d_precAdd_110) (coe v8) (coe v4))
                               v10)
                            (\ v11 ->
                               case coe v11 of
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                   -> coe
                                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                        (coe
                                           MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                           (coe d___64 () erased) erased erased
                                           (d_ppx_118
                                              (coe v0) (coe v1) (coe d_precAdd_110) (coe v9)
                                              (coe v4))
                                           (\ v14 ->
                                              coe
                                                MAlonzo.Code.Effect.Applicative.du_return_68
                                                (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                                   (coe d___64 () erased))
                                                (coe
                                                   d_pars_104
                                                   (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                                      (coe
                                                         MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                         (coe v2) (coe d_precAdd_110)))
                                                   (coe
                                                      MAlonzo.Code.Text.Printf.d_printf_26
                                                      ("%s + %s" :: Data.Text.Text) v13 v14))))
                                        v12
                                 _ -> MAlonzo.RTE.mazUnreachableError)))
             MAlonzo.Code.Lang.C_mul_192
               -> coe
                    MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
                    (coe
                       (\ v10 ->
                          coe
                            MAlonzo.Code.Function.Base.du__'8728''8242'__216
                            (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                            (\ v11 ->
                               MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v11))
                            (coe
                               MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                               (d_ppx_118 (coe v0) (coe v1) (coe d_precMul_112) (coe v8) (coe v4))
                               v10)
                            (\ v11 ->
                               case coe v11 of
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                   -> coe
                                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                        (coe
                                           MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                           (coe d___64 () erased) erased erased
                                           (d_ppx_118
                                              (coe v0) (coe v1) (coe d_precMul_112) (coe v9)
                                              (coe v4))
                                           (\ v14 ->
                                              coe
                                                MAlonzo.Code.Effect.Applicative.du_return_68
                                                (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                                   (coe d___64 () erased))
                                                (coe
                                                   d_pars_104
                                                   (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                                      (coe
                                                         MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                         (coe v2) (coe d_precMul_112)))
                                                   (coe
                                                      MAlonzo.Code.Text.Printf.d_printf_26
                                                      ("(%s * %s)" :: Data.Text.Text) v13 v14))))
                                        v12
                                 _ -> MAlonzo.RTE.mazUnreachableError)))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_scaledown_244 v7 v8
        -> coe
             MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
             (coe
                (\ v9 ->
                   coe
                     MAlonzo.Code.Function.Base.du__'8728''8242'__216
                     (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                     (\ v10 ->
                        MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v10))
                     (coe
                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                        (d_ppx_118
                           (coe v0) (coe v1)
                           (coe addInt (coe (1 :: Integer)) (coe d_precApp_114)) (coe v8)
                           (coe v4))
                        v9)
                     (\ v10 ->
                        case coe v10 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v11 v12
                            -> coe
                                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                 (coe
                                    MAlonzo.Code.Effect.Applicative.du_return_68
                                    (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                       (coe d___64 () erased))
                                    (coe
                                       d_pars_104
                                       (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                          (coe
                                             MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                             (coe v2) (coe d_precApp_114)))
                                       (coe
                                          MAlonzo.Code.Text.Printf.d_printf_26
                                          ("scaledown %u %s" :: Data.Text.Text) v7 v12)))
                                 v11
                          _ -> MAlonzo.RTE.mazUnreachableError)))
      MAlonzo.Code.Lang.C_let'8242'_246 v6 v8 v9
        -> coe
             MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
             (coe
                (\ v10 ->
                   coe
                     MAlonzo.Code.Function.Base.du__'8728''8242'__216
                     (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                     (\ v11 ->
                        MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v11))
                     (coe
                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                        d_fresh'45'var_96 v10)
                     (\ v11 ->
                        case coe v11 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                            -> coe
                                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                 (coe
                                    MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                    (coe d___64 () erased) erased erased
                                    (d_ppx_118
                                       (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v6))
                                       (coe addInt (coe (1 :: Integer)) (coe d_precLet_108))
                                       (coe v8) (coe v4))
                                    (\ v14 ->
                                       coe
                                         MAlonzo.Code.Effect.Monad.d__'62''62''61'__34
                                         (coe d___64 () erased) erased erased
                                         (d_ppx_118
                                            (coe
                                               MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                               (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)))
                                            (coe v1) (coe d_precLet_108) (coe v9)
                                            (coe
                                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                                               (coe v13)))
                                         (\ v15 ->
                                            coe
                                              MAlonzo.Code.Effect.Applicative.du_return_68
                                              (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                                 (coe d___64 () erased))
                                              (coe
                                                 d_pars_104
                                                 (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                                    (coe
                                                       MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                                       (coe v2) (coe d_precLet_108)))
                                                 (coe
                                                    MAlonzo.Code.Text.Printf.d_printf_26
                                                    ("let %s = %s in\n%s" :: Data.Text.Text) v13 v14
                                                    v15)))))
                                 v12
                          _ -> MAlonzo.RTE.mazUnreachableError)))
      MAlonzo.Code.Lang.C_un_248 v7 v8
        -> coe
             MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
             (coe
                (\ v9 ->
                   coe
                     MAlonzo.Code.Function.Base.du__'8728''8242'__216
                     (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
                     (\ v10 ->
                        MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20 (coe v10))
                     (coe
                        MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                        (d_ppx_118
                           (coe v0) (coe v1)
                           (coe addInt (coe (1 :: Integer)) (coe d_precApp_114)) (coe v8)
                           (coe v4))
                        v9)
                     (\ v10 ->
                        case coe v10 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v11 v12
                            -> coe
                                 MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                 (coe
                                    MAlonzo.Code.Effect.Applicative.du_return_68
                                    (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                       (coe d___64 () erased))
                                    (coe
                                       d_pars_104
                                       (MAlonzo.Code.Relation.Nullary.Decidable.Core.d_does_28
                                          (coe
                                             MAlonzo.Code.Data.Nat.Properties.d__'62''63'__3178
                                             (coe v2) (coe d_precApp_114)))
                                       (coe
                                          MAlonzo.Code.Text.Printf.d_printf_26
                                          ("(%s %s)" :: Data.Text.Text) (d_uop_102 (coe v7)) v12)))
                                 v11
                          _ -> MAlonzo.RTE.mazUnreachableError)))
      _ -> MAlonzo.RTE.mazUnreachableError
-- PP.pp
d_pp_320 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 ->
  AgdaAny ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58
d_pp_320 v0 v1
  = coe d_ppx_118 (coe v0) (coe v1) (coe (0 :: Integer))
