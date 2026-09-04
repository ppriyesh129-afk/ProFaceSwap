package com.profaceswap

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

class FaceSwapEngine(private val context: Context) {

    private val env = OrtEnvironment.getEnvironment()

    private var detectorSession: OrtSession? = null
    private var landmarkerSession: OrtSession? = null
    private var blendSwapSession: OrtSession? = null

    private var detector: BlazeFaceDetector? = null

    fun loadModels(): Boolean {
        return try {
            detectorSession = createSession("models/face_detection_short_range.onnx")
            landmarkerSession = createSession("models/face_landmarker_Nx3x256x256.onnx")
            blendSwapSession = createSession("models/blendswap_256.onnx")

            detector = BlazeFaceDetector(detectorSession!!)

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

    fun processSwap(target: Bitmap, source: Bitmap): Bitmap {
        detector?.detect(target)
        return target
    }

    fun close() {
        detectorSession?.close()
        landmarkerSession?.close()
        blendSwapSession?.close()
    }
}
