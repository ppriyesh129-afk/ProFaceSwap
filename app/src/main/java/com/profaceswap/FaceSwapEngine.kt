package com.profaceswap

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log

class FaceSwapEngine(
    private val context: Context
) {

    companion object {
        private const val TAG = "ProFaceSwapEngine"
    }

    private val environment =
        OrtEnvironment.getEnvironment()

    private var detectorSession: OrtSession? = null
    private var landmarkerSession: OrtSession? = null
    private var arcFaceSession: OrtSession? = null
    private var hyperSwapSession: OrtSession? = null

    fun loadModels(): Boolean {

        return try {

            Log.d(TAG, "Loading detector...")

            detectorSession =
                createSession(
                    "models/face_detection_short_range.onnx"
                )

            Log.d(
                TAG,
                ModelInspector.describe(
                    detectorSession!!
                )
            )

            Log.d(TAG, "Loading 478-point landmarker...")

            landmarkerSession =
                createSession(
                    "models/face_landmarker_Nx3x256x256.onnx"
                )

            Log.d(
                TAG,
                ModelInspector.describe(
                    landmarkerSession!!
                )
            )

            Log.d(TAG, "Loading ArcFace...")

            arcFaceSession =
                createSession(
                    "models/arcface_w600k_r50.onnx"
                )

            Log.d(
                TAG,
                ModelInspector.describe(
                    arcFaceSession!!
                )
            )

            Log.d(
                TAG,
                "HyperSwap will be inspected separately."
            )

            true

        } catch (e: Throwable) {

            Log.e(
                TAG,
                "MODEL LOAD ERROR",
                e
            )

            false
        }
    }

    fun loadHyperSwap(): Boolean {

        return try {

            Log.d(
                TAG,
                "Loading HyperSwap 1a..."
            )

            hyperSwapSession =
                createSession(
                    "models/hyperswap_1a_256.onnx"
                )

            Log.d(
                TAG,
                ModelInspector.describe(
                    hyperSwapSession!!
                )
            )

            true

        } catch (e: Throwable) {

            Log.e(
                TAG,
                "HYPERSWAP LOAD ERROR",
                e
            )

            false
        }
    }

    private fun createSession(
        assetPath: String
    ): OrtSession {

        val bytes =
            context.assets
                .open(assetPath)
                .use { input ->
                    input.readBytes()
                }

        return environment.createSession(
            bytes,
            OrtSession.SessionOptions()
        )
    }

    fun processSwap(
        target: Bitmap,
        source: Bitmap
    ): Bitmap {

        throw IllegalStateException(
            "HyperSwap diagnostic build: model inspection only"
        )
    }

    fun close() {

        try {
            hyperSwapSession?.close()
        } catch (_: Throwable) {
        }

        try {
            arcFaceSession?.close()
        } catch (_: Throwable) {
        }

        try {
            landmarkerSession?.close()
        } catch (_: Throwable) {
        }

        try {
            detectorSession?.close()
        } catch (_: Throwable) {
        }

        hyperSwapSession = null
        arcFaceSession = null
        landmarkerSession = null
        detectorSession = null
    }
}
