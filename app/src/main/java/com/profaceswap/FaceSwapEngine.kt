package com.profaceswap

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

class FaceSwapEngine(private val context: Context) {

    private val env = OrtEnvironment.getEnvironment()

    private var detector: OrtSession? = null
    private var landmarker: OrtSession? = null
    private var blendSwap: OrtSession? = null

    fun loadModels(): Boolean {
        return try {
            detector = createSession("models/face_detection_short_range.onnx")
            landmarker = createSession("models/face_landmarker_Nx3x256x256.onnx")
            blendSwap = createSession("models/blendswap_256.onnx")
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    private fun createSession(assetPath: String): OrtSession {
        val bytes = context.assets.open(assetPath).readBytes()
        return env.createSession(bytes)
    }

    fun processSwap(
        target: Bitmap,
        source: Bitmap
    ): Bitmap {
        // Real inference will be added next.
        return target
    }

    fun close() {
        detector?.close()
        landmarker?.close()
        blendSwap?.close()
    }
}
