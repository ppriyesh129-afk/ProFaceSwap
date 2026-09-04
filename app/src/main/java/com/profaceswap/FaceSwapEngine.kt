package com.profaceswap

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

class FaceSwapEngine(private val context: Context) {

    private val environment =
        OrtEnvironment.getEnvironment()

    private var detectorSession: OrtSession? = null
    private var landmarkerSession: OrtSession? = null
    private var blendSwapSession: OrtSession? = null

    private var detector: BlazeFaceDetector? = null
    private var landmarker: FaceLandmarker? = null

    fun loadModels(): Boolean {
        return try {

            detectorSession =
                createSession(
                    "models/face_detection_short_range.onnx"
                )

            landmarkerSession =
                createSession(
                    "models/face_landmarker_Nx3x256x256.onnx"
                )

            blendSwapSession =
                createSession(
                    "models/blendswap_256.onnx"
                )

            detector =
                BlazeFaceDetector(detectorSession!!)

            landmarker =
                FaceLandmarker(landmarkerSession!!)

            true

        } catch (e: Exception) {

            e.printStackTrace()
            false
        }
    }

    private fun createSession(
        assetPath: String
    ): OrtSession {

        val bytes =
            context.assets
                .open(assetPath)
                .use { it.readBytes() }

        return environment.createSession(bytes)
    }

    fun processSwap(
        target: Bitmap,
        source: Bitmap
    ): Bitmap {

        val targetFaces =
            detector?.detect(target)
                ?: emptyList()

        if (targetFaces.isEmpty()) {
            return target
        }

        val targetFace =
            targetFaces.maxByOrNull { it.score }
                ?: return target

        val left =
            targetFace.left.toInt().coerceIn(
                0,
                target.width - 1
            )

        val top =
            targetFace.top.toInt().coerceIn(
                0,
                target.height - 1
            )

        val right =
            targetFace.right.toInt().coerceIn(
                left + 1,
                target.width
            )

        val bottom =
            targetFace.bottom.toInt().coerceIn(
                top + 1,
                target.height
            )

        val faceCrop =
            Bitmap.createBitmap(
                target,
                left,
                top,
                right - left,
                bottom - top
            )

        val landmarks =
            landmarker?.detect(faceCrop)
                ?: emptyArray()

        faceCrop.recycle()

        return target
    }

    fun close() {

        detectorSession?.close()
        landmarkerSession?.close()
        blendSwapSession?.close()
    }
}
