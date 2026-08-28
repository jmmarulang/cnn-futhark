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

module MAlonzo.Code.Effect.Monad.State.Transformer where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.Sigma
import qualified MAlonzo.Code.Agda.Builtin.Unit
import qualified MAlonzo.Code.Agda.Primitive
import qualified MAlonzo.Code.Algebra.Bundles.Raw
import qualified MAlonzo.Code.Data.Product.Base
import qualified MAlonzo.Code.Effect.Applicative
import qualified MAlonzo.Code.Effect.Choice
import qualified MAlonzo.Code.Effect.Empty
import qualified MAlonzo.Code.Effect.Functor
import qualified MAlonzo.Code.Effect.Monad
import qualified MAlonzo.Code.Effect.Monad.Reader.Transformer.Base
import qualified MAlonzo.Code.Effect.Monad.State.Transformer.Base
import qualified MAlonzo.Code.Effect.Monad.Writer.Transformer.Base
import qualified MAlonzo.Code.Function.Base
import qualified MAlonzo.Code.Level

-- Effect.Monad.State.Transformer.functor
d_functor_24 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Effect.Functor.T_RawFunctor_24 ->
  MAlonzo.Code.Effect.Functor.T_RawFunctor_24
d_functor_24 ~v0 ~v1 ~v2 ~v3 v4 = du_functor_24 v4
du_functor_24 ::
  MAlonzo.Code.Effect.Functor.T_RawFunctor_24 ->
  MAlonzo.Code.Effect.Functor.T_RawFunctor_24
du_functor_24 v0
  = coe
      MAlonzo.Code.Effect.Functor.C_constructor_44
      (coe
         (\ v1 v2 v3 v4 ->
            coe
              MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
              (coe
                 (\ v5 ->
                    coe
                      MAlonzo.Code.Effect.Functor.d__'60''36''62'__30 v0 erased erased
                      (coe
                         MAlonzo.Code.Data.Product.Base.du_map'8322'_150 (coe (\ v6 -> v3)))
                      (coe
                         MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68 v4
                         v5)))))
-- Effect.Monad.State.Transformer.applicative
d_applicative_48 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Effect.Monad.T_RawMonad_24 ->
  MAlonzo.Code.Effect.Applicative.T_RawApplicative_20
d_applicative_48 ~v0 ~v1 ~v2 ~v3 v4 = du_applicative_48 v4
du_applicative_48 ::
  MAlonzo.Code.Effect.Monad.T_RawMonad_24 ->
  MAlonzo.Code.Effect.Applicative.T_RawApplicative_20
du_applicative_48 v0
  = coe
      MAlonzo.Code.Effect.Applicative.C_constructor_78
      (coe
         du_functor_24
         (coe
            MAlonzo.Code.Effect.Applicative.d_rawFunctor_30
            (coe MAlonzo.Code.Effect.Monad.d_rawApplicative_32 (coe v0))))
      (coe
         (\ v1 v2 ->
            coe
              MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
              (coe
                 MAlonzo.Code.Function.Base.du__'8728''8242'__216
                 (coe
                    MAlonzo.Code.Effect.Applicative.d_pure_32
                    (MAlonzo.Code.Effect.Monad.d_rawApplicative_32 (coe v0)) erased)
                 (coe
                    (\ v3 ->
                       coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3) (coe v2))))))
      (coe
         (\ v1 v2 v3 v4 ->
            coe
              MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
              (coe
                 (\ v5 ->
                    coe
                      MAlonzo.Code.Effect.Monad.d__'62''62''61'__34 v0 erased erased
                      (coe
                         MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68 v3
                         v5)
                      (\ v6 ->
                         case coe v6 of
                           MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v7 v8
                             -> coe
                                  MAlonzo.Code.Effect.Monad.d__'62''62''61'__34 v0 erased erased
                                  (coe
                                     MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                     v4 v7)
                                  (\ v9 ->
                                     case coe v9 of
                                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v10 v11
                                         -> coe
                                              MAlonzo.Code.Effect.Applicative.d_pure_32
                                              (MAlonzo.Code.Effect.Monad.d_rawApplicative_32
                                                 (coe v0))
                                              erased
                                              (coe
                                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                 (coe v10) (coe v8 v11))
                                       _ -> MAlonzo.RTE.mazUnreachableError)
                           _ -> MAlonzo.RTE.mazUnreachableError)))))
-- Effect.Monad.State.Transformer.empty
d_empty_130 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Effect.Empty.T_RawEmpty_16 ->
  MAlonzo.Code.Effect.Empty.T_RawEmpty_16
d_empty_130 ~v0 ~v1 ~v2 ~v3 v4 = du_empty_130 v4
du_empty_130 ::
  MAlonzo.Code.Effect.Empty.T_RawEmpty_16 ->
  MAlonzo.Code.Effect.Empty.T_RawEmpty_16
du_empty_130 v0
  = coe
      MAlonzo.Code.Effect.Empty.C_constructor_26
      (coe
         (\ v1 ->
            coe
              MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
              (let v2 = coe MAlonzo.Code.Effect.Empty.d_empty_22 v0 erased in
               coe (coe (\ v3 -> v2)))))
-- Effect.Monad.State.Transformer.choice
d_choice_134 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Effect.Choice.T_RawChoice_16 ->
  MAlonzo.Code.Effect.Choice.T_RawChoice_16
d_choice_134 ~v0 ~v1 ~v2 ~v3 v4 = du_choice_134 v4
du_choice_134 ::
  MAlonzo.Code.Effect.Choice.T_RawChoice_16 ->
  MAlonzo.Code.Effect.Choice.T_RawChoice_16
du_choice_134 v0
  = coe
      MAlonzo.Code.Effect.Choice.C_constructor_26
      (coe
         (\ v1 v2 v3 ->
            coe
              MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
              (coe
                 (\ v4 ->
                    coe
                      MAlonzo.Code.Effect.Choice.d__'60''124''62'__22 v0 erased
                      (coe
                         MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68 v2
                         v4)
                      (coe
                         MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68 v3
                         v4)))))
-- Effect.Monad.State.Transformer.applicativeZero
d_applicativeZero_154 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Effect.Monad.T_RawMonadZero_208 ->
  MAlonzo.Code.Effect.Applicative.T_RawApplicativeZero_122
d_applicativeZero_154 ~v0 ~v1 ~v2 ~v3 v4
  = du_applicativeZero_154 v4
du_applicativeZero_154 ::
  MAlonzo.Code.Effect.Monad.T_RawMonadZero_208 ->
  MAlonzo.Code.Effect.Applicative.T_RawApplicativeZero_122
du_applicativeZero_154 v0
  = coe
      MAlonzo.Code.Effect.Applicative.C_constructor_176
      (coe
         du_applicative_48
         (coe MAlonzo.Code.Effect.Monad.d_rawMonad_216 (coe v0)))
      (coe
         du_empty_130
         (coe MAlonzo.Code.Effect.Monad.d_rawEmpty_218 (coe v0)))
-- Effect.Monad.State.Transformer.alternative
d_alternative_158 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Effect.Monad.T_RawMonadPlus_288 ->
  MAlonzo.Code.Effect.Applicative.T_RawAlternative_184
d_alternative_158 ~v0 ~v1 ~v2 ~v3 v4 = du_alternative_158 v4
du_alternative_158 ::
  MAlonzo.Code.Effect.Monad.T_RawMonadPlus_288 ->
  MAlonzo.Code.Effect.Applicative.T_RawAlternative_184
du_alternative_158 v0
  = coe
      MAlonzo.Code.Effect.Applicative.C_constructor_246
      (coe
         du_applicativeZero_154
         (coe MAlonzo.Code.Effect.Monad.d_rawMonadZero_296 (coe v0)))
      (coe
         du_choice_134
         (coe MAlonzo.Code.Effect.Monad.d_rawChoice_298 (coe v0)))
-- Effect.Monad.State.Transformer.monad
d_monad_238 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Effect.Monad.T_RawMonad_24 ->
  MAlonzo.Code.Effect.Monad.T_RawMonad_24
d_monad_238 ~v0 ~v1 ~v2 ~v3 v4 = du_monad_238 v4
du_monad_238 ::
  MAlonzo.Code.Effect.Monad.T_RawMonad_24 ->
  MAlonzo.Code.Effect.Monad.T_RawMonad_24
du_monad_238 v0
  = coe
      MAlonzo.Code.Effect.Monad.C_constructor_98
      (coe du_applicative_48 (coe v0))
      (coe
         (\ v1 v2 v3 v4 ->
            coe
              MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
              (coe
                 (\ v5 ->
                    coe
                      MAlonzo.Code.Effect.Monad.d__'62''62''61'__34 v0 erased erased
                      (coe
                         MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68 v3
                         v5)
                      (\ v6 ->
                         case coe v6 of
                           MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v7 v8
                             -> coe
                                  MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68
                                  (coe v4 v8) v7
                           _ -> MAlonzo.RTE.mazUnreachableError)))))
-- Effect.Monad.State.Transformer.monadZero
d_monadZero_310 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Effect.Monad.T_RawMonadZero_208 ->
  MAlonzo.Code.Effect.Monad.T_RawMonadZero_208
d_monadZero_310 ~v0 ~v1 ~v2 ~v3 v4 = du_monadZero_310 v4
du_monadZero_310 ::
  MAlonzo.Code.Effect.Monad.T_RawMonadZero_208 ->
  MAlonzo.Code.Effect.Monad.T_RawMonadZero_208
du_monadZero_310 v0
  = coe
      MAlonzo.Code.Effect.Monad.C_constructor_280
      (coe
         du_monad_238
         (coe MAlonzo.Code.Effect.Monad.d_rawMonad_216 (coe v0)))
      (coe
         du_empty_130
         (coe MAlonzo.Code.Effect.Monad.d_rawEmpty_218 (coe v0)))
-- Effect.Monad.State.Transformer.monadPlus
d_monadPlus_314 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Effect.Monad.T_RawMonadPlus_288 ->
  MAlonzo.Code.Effect.Monad.T_RawMonadPlus_288
d_monadPlus_314 ~v0 ~v1 ~v2 ~v3 v4 = du_monadPlus_314 v4
du_monadPlus_314 ::
  MAlonzo.Code.Effect.Monad.T_RawMonadPlus_288 ->
  MAlonzo.Code.Effect.Monad.T_RawMonadPlus_288
du_monadPlus_314 v0
  = coe
      MAlonzo.Code.Effect.Monad.C_constructor_370
      (coe
         du_monadZero_310
         (coe MAlonzo.Code.Effect.Monad.d_rawMonadZero_296 (coe v0)))
      (coe
         du_choice_134
         (coe MAlonzo.Code.Effect.Monad.d_rawChoice_298 (coe v0)))
-- Effect.Monad.State.Transformer.monadT
d_monadT_394 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  (() -> ()) ->
  MAlonzo.Code.Effect.Monad.T_RawMonad_24 ->
  MAlonzo.Code.Effect.Monad.T_RawMonadTd_382
d_monadT_394 ~v0 ~v1 ~v2 ~v3 v4 = du_monadT_394 v4
du_monadT_394 ::
  MAlonzo.Code.Effect.Monad.T_RawMonad_24 ->
  MAlonzo.Code.Effect.Monad.T_RawMonadTd_382
du_monadT_394 v0
  = coe
      MAlonzo.Code.Effect.Monad.C_constructor_448
      (coe
         (\ v1 v2 ->
            coe
              MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
              (coe
                 (\ v3 ->
                    coe
                      MAlonzo.Code.Effect.Functor.d__'60''36''62'__30
                      (MAlonzo.Code.Effect.Applicative.d_rawFunctor_30
                         (coe MAlonzo.Code.Effect.Monad.d_rawApplicative_32 (coe v0)))
                      erased erased
                      (\ v4 ->
                         coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3) (coe v4))
                      v2))))
      (coe du_monad_238 (coe v0))
-- Effect.Monad.State.Transformer.monadState
d_monadState_460 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Effect.Monad.T_RawMonad_24 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28
d_monadState_460 ~v0 ~v1 ~v2 ~v3 v4 = du_monadState_460 v4
du_monadState_460 ::
  MAlonzo.Code.Effect.Monad.T_RawMonad_24 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28
du_monadState_460 v0
  = coe
      MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_constructor_46
      (coe
         (\ v1 v2 ->
            coe
              MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
              (coe
                 (\ v3 ->
                    coe
                      MAlonzo.Code.Effect.Applicative.d_pure_32
                      (MAlonzo.Code.Effect.Monad.d_rawApplicative_32 (coe v0)) erased
                      (coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3)
                         (coe v2 v3))))))
      (coe
         (\ v1 ->
            coe
              MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
              (coe
                 (\ v2 ->
                    coe
                      MAlonzo.Code.Effect.Applicative.d_pure_32
                      (MAlonzo.Code.Effect.Monad.d_rawApplicative_32 (coe v0)) erased
                      (coe
                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v1 v2)
                         (coe
                            MAlonzo.Code.Level.C_lift_20
                            (coe MAlonzo.Code.Agda.Builtin.Unit.C_tt_8)))))))
-- Effect.Monad.State.Transformer.liftStateT
d_liftStateT_528 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  () ->
  MAlonzo.Code.Effect.Monad.T_RawMonad_24 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28
d_liftStateT_528 ~v0 ~v1 ~v2 ~v3 ~v4 v5 v6
  = du_liftStateT_528 v5 v6
du_liftStateT_528 ::
  MAlonzo.Code.Effect.Monad.T_RawMonad_24 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28
du_liftStateT_528 v0 v1
  = coe
      MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_constructor_46
      (coe
         (\ v2 v3 ->
            coe
              MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
              (coe
                 (\ v4 ->
                    coe
                      MAlonzo.Code.Effect.Functor.d__'60''36''62'__30
                      (MAlonzo.Code.Effect.Applicative.d_rawFunctor_30
                         (coe MAlonzo.Code.Effect.Monad.d_rawApplicative_32 (coe v0)))
                      erased erased
                      (\ v5 ->
                         coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) (coe v5))
                      (coe
                         MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_gets_38 v1
                         erased v3)))))
      (coe
         (\ v2 ->
            coe
              MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_mkStateT_70
              (coe
                 (\ v3 ->
                    coe
                      MAlonzo.Code.Effect.Functor.d__'60''36''62'__30
                      (MAlonzo.Code.Effect.Applicative.d_rawFunctor_30
                         (coe MAlonzo.Code.Effect.Monad.d_rawApplicative_32 (coe v0)))
                      erased erased
                      (\ v4 ->
                         coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3) (coe v4))
                      (coe
                         MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_modify_40 v1
                         v2)))))
-- Effect.Monad.State.Transformer.liftReaderT
d_liftReaderT_556 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28
d_liftReaderT_556 ~v0 ~v1 ~v2 ~v3 ~v4 v5 = du_liftReaderT_556 v5
du_liftReaderT_556 ::
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28
du_liftReaderT_556 v0
  = coe
      MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_constructor_46
      (coe
         (\ v1 v2 ->
            coe
              MAlonzo.Code.Effect.Monad.Reader.Transformer.Base.C_mkReaderT_54
              (let v3
                     = coe
                         MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_gets_38 v0
                         erased v2 in
               coe (coe (\ v4 -> v3)))))
      (coe
         (\ v1 ->
            coe
              MAlonzo.Code.Effect.Monad.Reader.Transformer.Base.C_mkReaderT_54
              (let v2
                     = coe
                         MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_modify_40 v0
                         v1 in
               coe (coe (\ v3 -> v2)))))
-- Effect.Monad.State.Transformer.liftWriterT
d_liftWriterT_580 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  MAlonzo.Code.Algebra.Bundles.Raw.T_RawMonoid_74 ->
  MAlonzo.Code.Effect.Functor.T_RawFunctor_24 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28
d_liftWriterT_580 ~v0 ~v1 ~v2 ~v3 ~v4 ~v5 v6 v7
  = du_liftWriterT_580 v6 v7
du_liftWriterT_580 ::
  MAlonzo.Code.Effect.Functor.T_RawFunctor_24 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28
du_liftWriterT_580 v0 v1
  = coe
      MAlonzo.Code.Effect.Monad.State.Transformer.Base.C_constructor_46
      (coe
         (\ v2 v3 ->
            coe
              MAlonzo.Code.Effect.Monad.Writer.Transformer.Base.C_mkWriterT_136
              (coe
                 (\ v4 ->
                    coe
                      MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_gets_38 v1
                      erased
                      (coe
                         MAlonzo.Code.Function.Base.du__'8728''8242'__216
                         (coe
                            (\ v5 ->
                               coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) (coe v5)))
                         (coe v3))))))
      (coe
         (\ v2 ->
            coe
              MAlonzo.Code.Effect.Monad.Writer.Transformer.Base.C_mkWriterT_136
              (coe
                 (\ v3 ->
                    coe
                      MAlonzo.Code.Effect.Functor.d__'60''36''62'__30 v0 erased erased
                      (let v4
                             = coe
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v3)
                                 (coe
                                    MAlonzo.Code.Level.C_lift_20
                                    (coe MAlonzo.Code.Agda.Builtin.Unit.C_tt_8)) in
                       coe (\ v5 -> v4))
                      (coe
                         MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_modify_40 v1
                         v2)))))
