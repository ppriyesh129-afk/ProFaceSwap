package com.profaceswap

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import kotlin.math.max
import kotlin.math.min

class FaceSwapEngine(
    private val context: android.content.Context
) {

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
                BlazeFaceDetector(
                    detectorSession!!
                )

            landmarker =
                FaceLandmarker(
                    landmarkerSession!!
                )

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

        val targetLandmarks =
            detectLandmarks(target)

                ?: return target

        val sourceLandmarks =
            detectLandmarks(source)

                ?: return target

        val sourceAligned =
            FaceAlignment.align(
                bitmap = source,
                landmarks = sourceLandmarks,
                outputSize = 112,
                useArcFaceTemplate = true
            ) ?: return target

        val targetAligned =
            FaceAlignment.align(
                bitmap = target,
                landmarks = targetLandmarks,
                outputSize = 256,
                useArcFaceTemplate = false
            ) ?: return target

        val swappedFace =
            runBlendSwap(
                sourceAligned.bitmap,
                targetAligned.bitmap
            )

        sourceAligned.bitmap.recycle()
        targetAligned.bitmap.recycle()

        return pasteBack(
            target = target,
            swapped = swappedFace,
            targetLandmarks = targetLandmarks
        )
    }

    private fun detectLandmarks(
        bitmap: Bitmap
    ): Array<FloatArray>? {

        val detected =
            detector?.detect(bitmap)
                ?: return null

        if (detected.isEmpty()) {
            return null
        }

        val face =
            detected.maxByOrNull {
                it.score
            } ?: return null

        val left =
            face.left
                .toInt()
                .coerceIn(
                    0,
                    bitmap.width - 1
                )

        val top =
            face.top
                .toInt()
                .coerceIn(
                    0,
                    bitmap.height - 1
                )

        val right =
            face.right
                .toInt()
                .coerceIn(
                    left + 1,
                    bitmap.width
                )

        val bottom =
            face.bottom
                .toInt()
                .coerceIn(
                    top + 1,
                    bitmap.height
                )

        val width =
            right - left

        val height =
            bottom - top

        val size =
            max(width, height)

        val margin =
            (size * 0.25f).toInt()

        val cropSize =
            size + margin * 2

        val centerX =
            (left + right) / 2

        val centerY =
            (top + bottom) / 2

        val cropLeft =
            (centerX - cropSize / 2)
                .coerceIn(
                    0,
                    max(0, bitmap.width - cropSize)
                )

        val cropTop =
            (centerY - cropSize / 2)
                .coerceIn(
                    0,
                    max(0, bitmap.height - cropSize)
                )

        val actualWidth =
            min(
                cropSize,
                bitmap.width - cropLeft
            )

        val actualHeight =
            min(
                cropSize,
                bitmap.height - cropTop
            )

        if (actualWidth <= 1 || actualHeight <= 1) {
            return null
        }

        val crop =
            Bitmap.createBitmap(
                bitmap,
                cropLeft,
                cropTop,
                actualWidth,
                actualHeight
            )

        val landmarks =
            landmarker?.detect(crop)
                ?: emptyArray()

        crop.recycle()

        if (landmarks.size < 292) {
            return null
        }

        /*
         * FaceLandmarker returns coordinates
         * in its 256x256 crop.
         *
         * Convert them back to full-image pixels.
         */

        val result =
            Array(landmarks.size) {
                index ->

                val x =
                    cropLeft +
                            landmarks[index][0] *
                            actualWidth / 256f

                val y =
                    cropTop +
                            landmarks[index][1] *
                            actualHeight / 256f

                floatArrayOf(
                    x,
                    y,
                    landmarks[index][2]
                )
            }

        return result
    }

    private fun runBlendSwap(
        source: Bitmap,
        target: Bitmap
    ): Bitmap {

        val session =
            blendSwapSession
                ?: throw IllegalStateException(
                    "BlendSwap model is not loaded"
                )

        val sourceBuffer =
            bitmapToCHW(
                source,
                112
            )

        val targetBuffer =
            bitmapToCHW(
                target,
                256
            )

        val sourceTensor =
            OnnxTensor.createTensor(
                environment,
                sourceBuffer,
                longArrayOf(
                    1,
                    3,
                    112,
                    112
                )
            )

        val targetTensor =
            OnnxTensor.createTensor(
                environment,
                targetBuffer,
                longArrayOf(
                    1,
                    3,
                    256,
                    256
                )
            )

        val inputs =
            HashMap<String, OnnxTensor>()

        for (input in session.inputInfo) {

            when (input.key) {

                "source" ->
                    inputs["source"] =
                        sourceTensor

                "target" ->
                    inputs["target"] =
                        targetTensor
            }
        }

        if (
            !inputs.containsKey("source") ||
            !inputs.containsKey("target")
        ) {

            sourceTensor.close()
            targetTensor.close()

            throw IllegalStateException(
                "BlendSwap inputs were not found"
            )
        }

        val result =
            session.run(inputs)

        val output =
            result[0].value

        val bitmap =
            outputToBitmap(output)

        result.close()

        sourceTensor.close()
        targetTensor.close()

        return bitmap
    }

    private fun bitmapToCHW(
        bitmap: Bitmap,
        size: Int
    ): java.nio.FloatBuffer {

        val resized =
            if (
                bitmap.width != size ||
                bitmap.height != size
            ) {
                Bitmap.createScaledBitmap(
                    bitmap,
                    size,
                    size,
                    true
                )
            } else {
                bitmap
            }

        val pixels =
            IntArray(size * size)

        resized.getPixels(
            pixels,
            0,
            size,
            0,
            0,
            size,
            size
        )

        val buffer =
            java.nio.FloatBuffer.allocate(
                3 * size * size
            )

        for (channel in 0..2) {

            for (pixel in pixels) {

                val value =
                    when (channel) {

                        0 ->
                            (pixel shr 16) and 0xFF

                        1 ->
                            (pixel shr 8) and 0xFF

                        else ->
                            pixel and 0xFF
                    }

                buffer.put(
                    value / 255f
                )
            }
        }

        buffer.rewind()

        if (resized !== bitmap) {
            resized.recycle()
        }

        return buffer
    }

    private fun outputToBitmap(
        raw: Any
    ): Bitmap {

        val data =
            when (raw) {

                is Array<*> -> {

                    val first =
                        raw[0]

                    when (first) {

                        is Array<*> ->
                            first

                        else ->
                            raw
                    }
                }

                else ->
                    throw IllegalStateException(
                        "Unexpected BlendSwap output"
                    )
            }

        val output =
            Bitmap.createBitmap(
                256,
                256,
                Bitmap.Config.ARGB_8888
            )

        val pixels =
            IntArray(
                256 * 256
            )

        /*
         * ONNX output is expected to be:
         * [1, 3, 256, 256]
         */

        val channels =
            data as Array<*>

        val red =
            channels[0] as FloatArray

        val green =
            channels[1] as FloatArray

        val blue =
            channels[2] as FloatArray

        for (i in pixels.indices) {

            val r =
                (red[i] * 255f)
                    .toInt()
                    .coerceIn(0, 255)

            val g =
                (green[i] * 255f)
                    .toInt()
                    .coerceIn(0, 255)

            val b =
                (blue[i] * 255f)
                    .toInt()
                    .coerceIn(0, 255)

            pixels[i] =
                android.graphics.Color.rgb(
                    r,
                    g,
                    b
                )
        }

        output.setPixels(
            pixels,
            0,
            256,
            0,
            0,
            256,
            256
        )

        return output
    }

    private fun pasteBack(
        target: Bitmap,
        swapped: Bitmap,
        targetLandmarks: Array<FloatArray>
    ): Bitmap {

        val aligned =
            FaceAlignment.align(
                bitmap = target,
                landmarks = targetLandmarks,
                outputSize = 256,
                useArcFaceTemplate = false
            ) ?: return target

        val inverse =
            Matrix()

        if (!aligned.matrix.invert(inverse)) {
            aligned.bitmap.recycle()
            swapped.recycle()
            return target
        }

        val result =
            target.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        val canvas =
            Canvas(result)

        val mask =
            android.graphics.Bitmap.createBitmap(
                256,
                256,
                Bitmap.Config.ALPHA_8
            )

        val maskCanvas =
            Canvas(mask)

        val maskPaint =
            Paint(Paint.ANTI_ALIAS_FLAG)

        maskPaint.shader =
            android.graphics.RadialGradient(
                128f,
                128f,
                145f,
                255,
                0,
                android.graphics.Shader.TileMode.CLAMP
            )

        maskCanvas.drawRect(
            0f,
            0f,
            256f,
            256f,
            maskPaint
        )

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        paint.alpha = 255

        canvas.save()

        canvas.drawBitmap(
            swapped,
            inverse,
            paint
        )

        canvas.restore()

        aligned.bitmap.recycle()
        swapped.recycle()
        mask.recycle()

        return result
    }

    fun close() {

        detectorSession?.close()
        landmarkerSession?.close()
        blendSwapSession?.close()

        detectorSession = null
        landmarkerSession = null
        blendSwapSession = null
    }
}
