package com.profaceswap

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RadialGradient
import android.graphics.Shader
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class FaceSwapEngine(
    private val context: Context
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

            android.util.Log.d(
                "ProFaceSwap",
                ModelInspector.describe(
                    blendSwapSession!!
                )
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
                ?: throw IllegalStateException(
                    "No face found in target image"
                )

        val sourceLandmarks =
            detectLandmarks(source)
                ?: throw IllegalStateException(
                    "No face found in source image"
                )

        val sourceAligned =
            FaceAlignment.align(
                bitmap = source,
                landmarks = sourceLandmarks,
                outputSize = 112,
                useArcFaceTemplate = true
            )
                ?: throw IllegalStateException(
                    "Could not align source face"
                )

        val targetAligned =
            FaceAlignment.align(
                bitmap = target,
                landmarks = targetLandmarks,
                outputSize = 256,
                useArcFaceTemplate = false
            )
                ?: throw IllegalStateException(
                    "Could not align target face"
                )

        try {

            val swappedFace =
                runBlendSwap(
                    sourceAligned.bitmap,
                    targetAligned.bitmap
                )

            return pasteBack(
                target = target,
                swapped = swappedFace,
                targetLandmarks = targetLandmarks
            )

        } finally {

            sourceAligned.bitmap.recycle()
            targetAligned.bitmap.recycle()
        }
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
                    max(
                        0,
                        bitmap.width - cropSize
                    )
                )

        val cropTop =
            (centerY - cropSize / 2)
                .coerceIn(
                    0,
                    max(
                        0,
                        bitmap.height - cropSize
                    )
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

        if (
            actualWidth <= 1 ||
            actualHeight <= 1
        ) {
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
            try {
                landmarker?.detect(crop)
                    ?: emptyArray()
            } finally {
                crop.recycle()
            }

        if (landmarks.size < 478) {
            return null
        }

        return Array(
            landmarks.size
        ) { index ->

            val x =
                cropLeft +
                        landmarks[index][0] *
                        actualWidth /
                        256f

            val y =
                cropTop +
                        landmarks[index][1] *
                        actualHeight /
                        256f

            floatArrayOf(
                x,
                y,
                landmarks[index][2]
            )
        }
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

        val sourceTensor =
            createImageTensor(
                source,
                112
            )

        val targetTensor =
            createImageTensor(
                target,
                256
            )

        try {

            val inputs =
                HashMap<String, OnnxTensor>()

            for (name in session.inputNames) {

                when (name) {

                    "source" ->
                        inputs[name] =
                            sourceTensor

                    "target" ->
                        inputs[name] =
                            targetTensor
                }
            }

            if (
                !inputs.containsKey("source") ||
                !inputs.containsKey("target")
            ) {

                throw IllegalStateException(
                    "BlendSwap requires source and target inputs"
                )
            }

            val result =
                session.run(inputs)

            try {

                val output =
                    result[0].value

                return decodeBlendSwapOutput(
                    output
                )

            } finally {

                result.close()
            }

        } finally {

            sourceTensor.close()
            targetTensor.close()
        }
    }

    private fun createImageTensor(
        bitmap: Bitmap,
        size: Int
    ): OnnxTensor {

        val pixels =
            getPixels(
                bitmap,
                size
            )

        val buffer =
            FloatBuffer.allocate(
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

        return OnnxTensor.createTensor(
            environment,
            buffer,
            longArrayOf(
                1,
                3,
                size.toLong(),
                size.toLong()
            )
        )
    }

    private fun getPixels(
        bitmap: Bitmap,
        size: Int
    ): IntArray {

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
            IntArray(
                size * size
            )

        resized.getPixels(
            pixels,
            0,
            size,
            0,
            0,
            size,
            size
        )

        if (resized !== bitmap) {
            resized.recycle()
        }

        return pixels
    }

    private fun decodeBlendSwapOutput(
        raw: Any
    ): Bitmap {

        val tensor =
            raw as? Array<*>
                ?: throw IllegalStateException(
                    "Unexpected BlendSwap output type: ${raw::class.java.name}"
                )

        if (tensor.isEmpty()) {
            throw IllegalStateException(
                "BlendSwap returned empty output"
            )
        }

        val batch =
            tensor[0] as? Array<*>
                ?: throw IllegalStateException(
                    "Unexpected BlendSwap output shape"
                )

        if (batch.size != 3) {
            throw IllegalStateException(
                "BlendSwap output has ${batch.size} channels, expected 3"
            )
        }

        val red =
            batch[0] as? FloatArray
                ?: throw IllegalStateException(
                    "Invalid red channel"
                )

        val green =
            batch[1] as? FloatArray
                ?: throw IllegalStateException(
                    "Invalid green channel"
                )

        val blue =
            batch[2] as? FloatArray
                ?: throw IllegalStateException(
                    "Invalid blue channel"
                )

        val expected =
            256 * 256

        if (
            red.size != expected ||
            green.size != expected ||
            blue.size != expected
        ) {
            throw IllegalStateException(
                "Unexpected BlendSwap output size: " +
                        "${red.size}, ${green.size}, ${blue.size}"
            )
        }

        val output =
            Bitmap.createBitmap(
                256,
                256,
                Bitmap.Config.ARGB_8888
            )

        val pixels =
            IntArray(expected)

        for (i in pixels.indices) {

            val r =
                (red[i] * 255f)
                    .roundToByte()

            val g =
                (green[i] * 255f)
                    .roundToByte()

            val b =
                (blue[i] * 255f)
                    .roundToByte()

            pixels[i] =
                Color.rgb(
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

    private fun Float.roundToByte(): Int {

        return this
            .coerceIn(0f, 1f)
            .times(255f)
            .toInt()
            .coerceIn(0, 255)
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
            )
                ?: return target

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

        val mask =
            createFeatherMask()

        val transformedFace =
            Bitmap.createBitmap(
                target.width,
                target.height,
                Bitmap.Config.ARGB_8888
            )

        val faceCanvas =
            Canvas(transformedFace)

        val facePaint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        faceCanvas.drawBitmap(
            swapped,
            inverse,
            facePaint
        )

        val transformedMask =
            Bitmap.createBitmap(
                target.width,
                target.height,
                Bitmap.Config.ALPHA_8
            )

        val maskCanvas =
            Canvas(transformedMask)

        maskCanvas.drawBitmap(
            mask,
            inverse,
            null
        )

        val resultCanvas =
            Canvas(result)

        val blendPaint =
            Paint(
                Paint.ANTI_ALIAS_FLAG
            )

        blendPaint.xfermode =
            PorterDuffXfermode(
                PorterDuff.Mode.DST_IN
            )

        /*
         * Keep the transformed face only
         * where the feathered mask exists.
         */

        val clippedFace =
            Bitmap.createBitmap(
                target.width,
                target.height,
                Bitmap.Config.ARGB_8888
            )

        val clippedCanvas =
            Canvas(clippedFace)

        clippedCanvas.drawBitmap(
            transformedFace,
            0f,
            0f,
            null
        )

        blendPaint.xfermode =
            PorterDuffXfermode(
                PorterDuff.Mode.DST_IN
            )

        clippedCanvas.drawBitmap(
            transformedMask,
            0f,
            0f,
            blendPaint
        )

        blendPaint.xfermode = null

        resultCanvas.drawBitmap(
            clippedFace,
            0f,
            0f,
            null
        )

        aligned.bitmap.recycle()
        swapped.recycle()
        mask.recycle()
        transformedFace.recycle()
        transformedMask.recycle()
        clippedFace.recycle()

        return result
    }

    private fun createFeatherMask(): Bitmap {

        val mask =
            Bitmap.createBitmap(
                256,
                256,
                Bitmap.Config.ALPHA_8
            )

        val canvas =
            Canvas(mask)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG
            )

        paint.shader =
            RadialGradient(
                128f,
                128f,
                128f,
                intArrayOf(
                    Color.WHITE,
                    Color.WHITE,
                    Color.TRANSPARENT
                ),
                floatArrayOf(
                    0f,
                    0.62f,
                    1f
                ),
                Shader.TileMode.CLAMP
            )

        canvas.drawRect(
            0f,
            0f,
            256f,
            256f,
            paint
        )

        return mask
    }

    fun close() {

        detectorSession?.close()
        landmarkerSession?.close()
        blendSwapSession?.close()

        detectorSession = null
        landmarkerSession = null
        blendSwapSession = null

        detector = null
        landmarker = null
    }
}
