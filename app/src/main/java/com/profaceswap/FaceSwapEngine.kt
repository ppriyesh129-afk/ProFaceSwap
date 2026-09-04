package com.profaceswap

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.Shader
import android.graphics.RadialGradient
import java.nio.FloatBuffer
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

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
            detectLandmarks(
                target
            ) ?: return target

        val sourceLandmarks =
            detectLandmarks(
                source
            ) ?: return target

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
            ) ?: run {
                sourceAligned.bitmap.recycle()
                return target
            }

        val swappedFace =
            try {

                runBlendSwap(
                    sourceAligned.bitmap,
                    targetAligned.bitmap
                )

            } catch (e: Exception) {

                sourceAligned.bitmap.recycle()
                targetAligned.bitmap.recycle()

                throw e
            }

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

        val boxWidth =
            right - left

        val boxHeight =
            bottom - top

        val longSide =
            max(
                boxWidth,
                boxHeight
            )

        /*
         * MediaPipe-style ROI:
         *
         * detector box expanded by 25% on
         * every side = 1.5x total size.
         */
        val cropSize =
            (longSide * 1.5f)
                .toInt()
                .coerceAtLeast(2)

        val centerX =
            (left + right) / 2f

        val centerY =
            (top + bottom) / 2f

        val cropLeft =
            (centerX - cropSize / 2f)
                .toInt()

        val cropTop =
            (centerY - cropSize / 2f)
                .toInt()

        val square =
            createSquareCrop(
                bitmap,
                cropLeft,
                cropTop,
                cropSize
            )

        /*
         * First landmark pass gives us the eye
         * positions needed to estimate roll.
         */
        val firstLandmarks =
            landmarker?.detect(square.bitmap)
                ?: emptyArray()

        if (firstLandmarks.size < 292) {
            square.bitmap.recycle()
            return null
        }

        val leftEye =
            averageLandmarks(
                firstLandmarks,
                intArrayOf(
                    33,
                    133,
                    160,
                    159,
                    158,
                    157,
                    173
                )
            )

        val rightEye =
            averageLandmarks(
                firstLandmarks,
                intArrayOf(
                    263,
                    362,
                    387,
                    386,
                    385,
                    384,
                    398
                )
            )

        /*
         * Landmark coordinates are in the 256x256
         * landmarker input coordinate system.
         */
        val dx =
            rightEye[0] - leftEye[0]

        val dy =
            rightEye[1] - leftEye[1]

        val angleDegrees =
            Math.toDegrees(
                atan2(
                    dy.toDouble(),
                    dx.toDouble()
                )
            ).toFloat()

        /*
         * Rotate only when meaningful roll exists.
         */
        if (kotlin.math.abs(angleDegrees) < 1.0f) {

            square.bitmap.recycle()

            return mapLandmarksToOriginal(
                firstLandmarks,
                cropLeft,
                cropTop,
                cropSize,
                cropSize
            )
        }

        val rotated =
            rotateBitmap(
                square.bitmap,
                -angleDegrees
            )

        square.bitmap.recycle()

        /*
         * Second landmark pass on the corrected ROI.
         */
        val finalLandmarks =
            landmarker?.detect(rotated.bitmap)
                ?: emptyArray()

        if (finalLandmarks.size < 292) {
            rotated.bitmap.recycle()
            return null
        }

        /*
         * rotated.bitmap is still square and has the
         * same dimensions as the original crop.
         *
         * Convert the landmark positions back through
         * the inverse rotation, then into original image
         * coordinates.
         */
        val mapped =
            mapRotatedLandmarksToOriginal(
                landmarks = finalLandmarks,
                cropLeft = cropLeft,
                cropTop = cropTop,
                cropSize = cropSize,
                angleDegrees = -angleDegrees
            )

        rotated.bitmap.recycle()

        return mapped
    }

    private data class SquareCrop(
        val bitmap: Bitmap,
        val left: Int,
        val top: Int
    )

    private fun createSquareCrop(
        bitmap: Bitmap,
        requestedLeft: Int,
        requestedTop: Int,
        requestedSize: Int
    ): SquareCrop {

        val size =
            requestedSize
                .coerceAtMost(
                    max(
                        bitmap.width,
                        bitmap.height
                    )
                )

        val left =
            requestedLeft
                .coerceIn(
                    0,
                    max(
                        0,
                        bitmap.width - size
                    )
                )

        val top =
            requestedTop
                .coerceIn(
                    0,
                    max(
                        0,
                        bitmap.height - size
                    )
                )

        val width =
            min(
                size,
                bitmap.width - left
            )

        val height =
            min(
                size,
                bitmap.height - top
            )

        val actualSize =
            min(
                width,
                height
            )

        val crop =
            Bitmap.createBitmap(
                bitmap,
                left,
                top,
                actualSize,
                actualSize
            )

        return SquareCrop(
            bitmap = crop,
            left = left,
            top = top
        )
    }

    private fun rotateBitmap(
        bitmap: Bitmap,
        degrees: Float
    ): SquareCrop {

        val matrix =
            Matrix().apply {
                postRotate(
                    degrees,
                    bitmap.width / 2f,
                    bitmap.height / 2f
                )
            }

        val rotated =
            Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.width,
                bitmap.height,
                matrix,
                true
            )

        return SquareCrop(
            bitmap = rotated,
            left = 0,
            top = 0
        )
    }

    private fun averageLandmarks(
        landmarks: Array<FloatArray>,
        indices: IntArray
    ): FloatArray {

        var x = 0f
        var y = 0f
        var count = 0

        for (index in indices) {

            if (
                index >= 0 &&
                index < landmarks.size &&
                landmarks[index].size >= 2
            ) {

                x += landmarks[index][0]
                y += landmarks[index][1]

                count++
            }
        }

        if (count == 0) {
            return floatArrayOf(
                128f,
                128f
            )
        }

        return floatArrayOf(
            x / count,
            y / count
        )
    }

    private fun mapLandmarksToOriginal(
        landmarks: Array<FloatArray>,
        cropLeft: Int,
        cropTop: Int,
        cropWidth: Int,
        cropHeight: Int
    ): Array<FloatArray> {

        return Array(
            landmarks.size
        ) { index ->

            val point =
                landmarks[index]

            val x =
                cropLeft +
                        point[0] *
                        cropWidth /
                        256f

            val y =
                cropTop +
                        point[1] *
                        cropHeight /
                        256f

            floatArrayOf(
                x,
                y,
                if (point.size >= 3) {
                    point[2]
                } else {
                    0f
                }
            )
        }
    }

    private fun mapRotatedLandmarksToOriginal(
        landmarks: Array<FloatArray>,
        cropLeft: Int,
        cropTop: Int,
        cropSize: Int,
        angleDegrees: Float
    ): Array<FloatArray> {

        val center =
            cropSize / 2f

        val radians =
            Math.toRadians(
                angleDegrees.toDouble()
            )

        val cosValue =
            cos(radians)

        val sinValue =
            sin(radians)

        return Array(
            landmarks.size
        ) { index ->

            val point =
                landmarks[index]

            val x =
                point[0] *
                        cropSize /
                        256f

            val y =
                point[1] *
                        cropSize /
                        256f

            val dx =
                x - center

            val dy =
                y - center

            /*
             * Inverse rotation.
             */
            val originalX =
                center +
                        (
                            dx * cosValue -
                                    dy * sinValue
                            ).toFloat()

            val originalY =
                center +
                        (
                            dx * sinValue +
                                    dy * cosValue
                            ).toFloat()

            floatArrayOf(
                cropLeft + originalX,
                cropTop + originalY,
                if (point.size >= 3) {
                    point[2]
                } else {
                    0f
                }
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

        for (input in session.inputInfo.keys) {

            when (input) {

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
            try {

                session.run(
                    inputs
                )

            } catch (e: Exception) {

                sourceTensor.close()
                targetTensor.close()

                throw e
            }

        try {

            val rawOutput =
                result[0].value

            return decodeBlendSwapOutput(
                rawOutput
            )

        } finally {

            result.close()

            sourceTensor.close()
            targetTensor.close()
        }
    }

    private fun bitmapToCHW(
        bitmap: Bitmap,
        size: Int
    ): FloatBuffer {

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

        val buffer =
            FloatBuffer.allocate(
                3 *
                        size *
                        size
            )

        /*
         * BlendSwap:
         * RGB
         * CHW
         * [0,1]
         */
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

    private fun decodeBlendSwapOutput(
        raw: Any
    ): Bitmap {

        /*
         * Expected BlendSwap output:
         *
         * [1, 3, 256, 256]
         *
         * We accept the normal nested ONNX Java
         * representation:
         *
         * Array -> Array -> Array -> FloatArray
         */
        val batch =
            raw as? Array<*>
                ?: throw IllegalStateException(
                    "Unexpected BlendSwap output type: ${raw::class.java.name}"
                )

        val channelsObject =
            batch.firstOrNull()
                as? Array<*>
                ?: throw IllegalStateException(
                    "Unexpected BlendSwap batch output"
                )

        if (channelsObject.size < 3) {

            throw IllegalStateException(
                "BlendSwap output has fewer than 3 channels"
            )
        }

        val channels =
            Array(
                3
            ) { channelIndex ->

                flattenFloatArray(
                    channelsObject[channelIndex]
                )
            }

        val expected =
            256 * 256

        for (channel in channels) {

            if (channel.size < expected) {

                throw IllegalStateException(
                    "BlendSwap output channel is too small: ${channel.size}"
                )
            }
        }

        val bitmap =
            Bitmap.createBitmap(
                256,
                256,
                Bitmap.Config.ARGB_8888
            )

        val pixels =
            IntArray(
                expected
            )

        for (i in pixels.indices) {

            val r =
                (channels[0][i] * 255f)
                    .toInt()
                    .coerceIn(
                        0,
                        255
                    )

            val g =
                (channels[1][i] * 255f)
                    .toInt()
                    .coerceIn(
                        0,
                        255
                    )

            val b =
                (channels[2][i] * 255f)
                    .toInt()
                    .coerceIn(
                        0,
                        255
                    )

            pixels[i] =
                Color.rgb(
                    r,
                    g,
                    b
                )
        }

        bitmap.setPixels(
            pixels,
            0,
            256,
            0,
            0,
            256,
            256
        )

        return bitmap
    }

    private fun flattenFloatArray(
        raw: Any?
    ): FloatArray {

        return when (raw) {

            is FloatArray ->
                raw

            is DoubleArray ->
                FloatArray(
                    raw.size
                ) {
                    raw[it].toFloat()
                }

            is Array<*> -> {

                val output =
                    ArrayList<Float>()

                fun collect(
                    value: Any?
                ) {

                    when (value) {

                        is FloatArray -> {

                            for (v in value) {
                                output.add(v)
                            }
                        }

                        is DoubleArray -> {

                            for (v in value) {
                                output.add(
                                    v.toFloat()
                                )
                            }
                        }

                        is Array<*> -> {

                            for (item in value) {
                                collect(item)
                            }
                        }
                    }
                }

                collect(raw)

                FloatArray(
                    output.size
                ) {
                    output[it]
                }
            }

            else ->
                throw IllegalStateException(
                    "Unexpected BlendSwap channel type"
                )
        }
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
            ) ?: run {
                swapped.recycle()
                return target
            }

        val inverse =
            Matrix()

        if (
            !aligned.matrix.invert(
                inverse
            )
        ) {

            aligned.bitmap.recycle()
            swapped.recycle()

            return target
        }

        /*
         * Create a soft face-shaped mask.
         */
        val mask =
            Bitmap.createBitmap(
                256,
                256,
                Bitmap.Config.ARGB_8888
            )

        val maskCanvas =
            Canvas(mask)

        val maskPaint =
            Paint(
                Paint.ANTI_ALIAS_FLAG
            )

        maskPaint.shader =
            RadialGradient(
                128f,
                128f,
                145f,
                intArrayOf(
                    Color.WHITE,
                    Color.WHITE,
                    Color.TRANSPARENT
                ),
                floatArrayOf(
                    0f,
                    0.68f,
                    1f
                ),
                Shader.TileMode.CLAMP
            )

        maskCanvas.drawRect(
            0f,
            0f,
            256f,
            256f,
            maskPaint
        )

        /*
         * Apply the mask to the generated face
         * before transforming it back.
         */
        val masked =
            Bitmap.createBitmap(
                256,
                256,
                Bitmap.Config.ARGB_8888
            )

        val maskedCanvas =
            Canvas(masked)

        maskedCanvas.drawBitmap(
            swapped,
            0f,
            0f,
            null
        )

        val maskApplyPaint =
            Paint(
                Paint.ANTI_ALIAS_FLAG
            )

        maskApplyPaint.xfermode =
            PorterDuffXfermode(
                PorterDuff.Mode.DST_IN
            )

        maskedCanvas.drawBitmap(
            mask,
            0f,
            0f,
            maskApplyPaint
        )

        maskApplyPaint.xfermode = null

        /*
         * Paste the masked result back into the
         * original full-resolution target.
         */
        val result =
            target.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        val resultCanvas =
            Canvas(result)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        resultCanvas.drawBitmap(
            masked,
            inverse,
            paint
        )

        aligned.bitmap.recycle()
        swapped.recycle()
        masked.recycle()
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

        detector = null
        landmarker = null
    }
}
