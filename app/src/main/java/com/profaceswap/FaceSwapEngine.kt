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
import android.graphics.RectF
import android.util.Log
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.sqrt

class FaceSwapEngine(
    private val context: Context
) {

    companion object {
        private const val TAG = "ProFaceSwapEngine"

        private const val ARC_SIZE = 112
        private const val HYPER_SIZE = 256
    }

    private val environment =
        OrtEnvironment.getEnvironment()

    private var detectorSession: OrtSession? = null
    private var landmarkerSession: OrtSession? = null
    private var arcFaceSession: OrtSession? = null
    private var hyperSwapSession: OrtSession? = null

    fun loadModels(): Boolean {

        return try {

            Log.d(TAG, "Loading BlazeFace")

            detectorSession =
                createSession(
                    "models/face_detection_short_range.onnx"
                )

            Log.d(TAG, "Loading 478-point landmarker")

            landmarkerSession =
                createSession(
                    "models/face_landmarker_Nx3x256x256.onnx"
                )

            Log.d(TAG, "Loading ArcFace")

            arcFaceSession =
                createSession(
                    "models/arcface_w600k_r50.onnx"
                )

            Log.d(TAG, "Basic models loaded")

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

            if (hyperSwapSession != null) {
                return true
            }

            Log.d(TAG, "Loading HyperSwap 1a")

            hyperSwapSession =
                createSession(
                    "models/hyperswap_1a_256.onnx"
                )

            Log.d(TAG, "HyperSwap loaded")

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

    fun processSwap(
        target: Bitmap,
        source: Bitmap
    ): Bitmap {

        val detector =
            detectorSession
                ?: throw IllegalStateException(
                    "BlazeFace is not loaded"
                )

        val landmarker =
            landmarkerSession
                ?: throw IllegalStateException(
                    "Face Landmarker is not loaded"
                )

        val arcFace =
            arcFaceSession
                ?: throw IllegalStateException(
                    "ArcFace is not loaded"
                )

        if (!loadHyperSwap()) {
            throw IllegalStateException(
                "HyperSwap could not be loaded"
            )
        }

        val hyperSwap =
            hyperSwapSession
                ?: throw IllegalStateException(
                    "HyperSwap is not loaded"
                )

        Log.d(TAG, "Detecting source face")

        val faceDetector =
            BlazeFaceDetector(detector)

        val sourceFaces =
            faceDetector.detect(source)

        if (sourceFaces.isEmpty()) {
            throw IllegalStateException(
                "No face found in source image"
            )
        }

        Log.d(TAG, "Detecting target face")

        val targetFaces =
            faceDetector.detect(target)

        if (targetFaces.isEmpty()) {
            throw IllegalStateException(
                "No face found in target image"
            )
        }

        val sourceFace =
            largestFace(sourceFaces)

        val targetFace =
            largestFace(targetFaces)

        val sourceAligned =
            prepareFace(
                source,
                sourceFace,
                landmarker
            )

        val targetAligned =
            prepareFace(
                target,
                targetFace,
                landmarker
            )

        try {

            Log.d(TAG, "Creating ArcFace embedding")

            val sourceEmbedding =
                createArcFaceEmbedding(
                    sourceAligned.bitmap,
                    arcFace
                )

            Log.d(TAG, "Running HyperSwap")

            val swap =
                runHyperSwap(
                    sourceEmbedding,
                    targetAligned.bitmap,
                    hyperSwap
                )

            Log.d(TAG, "Pasting result")

            return pasteBack(
                original = target,
                swap = swap,
                transform = targetAligned.transform
            )

        } finally {

            sourceAligned.bitmap.recycle()
            targetAligned.bitmap.recycle()
        }
    }

    private fun prepareFace(
        bitmap: Bitmap,
        detection: BlazeFaceResult,
        landmarkerSession: OrtSession
    ): PreparedFace {

        val faceLandmarker =
            FaceLandmarker(
                landmarkerSession
            )

        val width =
            detection.right - detection.left

        val height =
            detection.bottom - detection.top

        val faceSize =
            max(width, height)

        if (faceSize <= 1f) {
            throw IllegalStateException(
                "Invalid face size"
            )
        }

        /*
         * The face-mesh ONNX pipeline expects a square face ROI.
         * We use a 1.5x face box, matching the practical
         * margin used by the 478-point pipeline.
         */

        val roiSize =
            faceSize * 1.5f

        val centerX =
            (detection.left + detection.right) * 0.5f

        val centerY =
            (detection.top + detection.bottom) * 0.5f

        var roiLeft =
            centerX - roiSize * 0.5f

        var roiTop =
            centerY - roiSize * 0.5f

        var roiRight =
            centerX + roiSize * 0.5f

        var roiBottom =
            centerY + roiSize * 0.5f

        if (roiLeft < 0f) {
            roiRight -= roiLeft
            roiLeft = 0f
        }

        if (roiTop < 0f) {
            roiBottom -= roiTop
            roiTop = 0f
        }

        if (roiRight > bitmap.width) {
            val difference =
                roiRight - bitmap.width

            roiLeft -= difference
            roiRight = bitmap.width
        }

        if (roiBottom > bitmap.height) {
            val difference =
                roiBottom - bitmap.height

            roiTop -= difference
            roiBottom = bitmap.height
        }

        roiLeft =
            roiLeft.coerceIn(
                0f,
                bitmap.width.toFloat() - 1f
            )

        roiTop =
            roiTop.coerceIn(
                0f,
                bitmap.height.toFloat() - 1f
            )

        roiRight =
            roiRight.coerceIn(
                roiLeft + 1f,
                bitmap.width.toFloat()
            )

        roiBottom =
            roiBottom.coerceIn(
                roiTop + 1f,
                bitmap.height.toFloat()
            )

        val cropLeft =
            roiLeft.toInt()

        val cropTop =
            roiTop.toInt()

        val cropWidth =
            max(
                1,
                (roiRight - roiLeft).toInt()
            )

        val cropHeight =
            max(
                1,
                (roiBottom - roiTop).toInt()
            )

        val crop =
            Bitmap.createBitmap(
                bitmap,
                cropLeft,
                cropTop,
                cropWidth.coerceAtMost(
                    bitmap.width - cropLeft
                ),
                cropHeight.coerceAtMost(
                    bitmap.height - cropTop
                )
            )

        val landmarks =
            faceLandmarker.detect(crop)

        if (landmarks.size < 292) {

            crop.recycle()

            throw IllegalStateException(
                "Could not obtain 478 face landmarks"
            )
        }

        /*
         * HyperSwap uses an ArcFace-style 5-point template.
         *
         * The same ArcFace 112 template is scaled to 256
         * for the HyperSwap target.
         */

        val aligned =
            alignToArcFaceTemplate(
                crop,
                landmarks,
                HYPER_SIZE
            )

        crop.recycle()

        val originalTransform =
            Matrix()

        originalTransform.setTranslate(
            cropLeft.toFloat(),
            cropTop.toFloat()
        )

        val inverse =
            Matrix()

        if (!aligned.matrix.invert(inverse)) {

            aligned.bitmap.recycle()

            throw IllegalStateException(
                "Could not invert face alignment"
            )
        }

        val finalTransform =
            Matrix()

        Matrix.setConcat(
            originalTransform,
            inverse,
            finalTransform
        )

        return PreparedFace(
            bitmap = aligned.bitmap,
            transform = finalTransform
        )
    }

    private fun alignToArcFaceTemplate(
        bitmap: Bitmap,
        landmarks: Array<FloatArray>,
        outputSize: Int
    ): FaceAlignment.AlignedFace {

        val source =
            FaceAlignment.fivePoints(
                landmarks
            )

        /*
         * arcface_112_v2 template.
         * HyperSwap uses the ArcFace 128-style alignment.
         * Scaling the canonical ArcFace template gives the
         * corresponding geometry at 256x256.
         */

        val template112 =
            arrayOf(
                floatArrayOf(
                    0.34191607f * 112f,
                    0.46157411f * 112f
                ),
                floatArrayOf(
                    0.65653393f * 112f,
                    0.45983393f * 112f
                ),
                floatArrayOf(
                    0.50022500f * 112f,
                    0.64050536f * 112f
                ),
                floatArrayOf(
                    0.37097589f * 112f,
                    0.82469196f * 112f
                ),
                floatArrayOf(
                    0.63151696f * 112f,
                    0.82325089f * 112f
                )
            )

        val scale =
            outputSize.toFloat() / 112f

        val template =
            Array(5) { index ->

                floatArrayOf(
                    template112[index][0] * scale,
                    template112[index][1] * scale
                )
            }

        val matrix =
            calculateSimilarityTransform(
                source,
                template
            )

        val output =
            Bitmap.createBitmap(
                outputSize,
                outputSize,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            Canvas(output)

        canvas.drawColor(
            Color.BLACK
        )

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        canvas.drawBitmap(
            bitmap,
            matrix,
            paint
        )

        return FaceAlignment.AlignedFace(
            bitmap = output,
            matrix = matrix
        )
    }

    private fun calculateSimilarityTransform(
        source: Array<FloatArray>,
        target: Array<FloatArray>
    ): Matrix {

        var sourceCx = 0.0
        var sourceCy = 0.0
        var targetCx = 0.0
        var targetCy = 0.0

        for (i in 0 until 5) {

            sourceCx +=
                source[i][0].toDouble()

            sourceCy +=
                source[i][1].toDouble()

            targetCx +=
                target[i][0].toDouble()

            targetCy +=
                target[i][1].toDouble()
        }

        sourceCx /= 5.0
        sourceCy /= 5.0

        targetCx /= 5.0
        targetCy /= 5.0

        var a = 0.0
        var b = 0.0
        var denominator = 0.0

        for (i in 0 until 5) {

            val sx =
                source[i][0].toDouble() -
                        sourceCx

            val sy =
                source[i][1].toDouble() -
                        sourceCy

            val tx =
                target[i][0].toDouble() -
                        targetCx

            val ty =
                target[i][1].toDouble() -
                        targetCy

            a +=
                sx * tx +
                        sy * ty

            b +=
                sx * ty -
                        sy * tx

            denominator +=
                sx * sx +
                        sy * sy
        }

        if (denominator < 0.000001) {

            throw IllegalStateException(
                "Face alignment failed"
            )
        }

        val magnitude =
            sqrt(
                a * a +
                        b * b
            )

        if (magnitude < 0.000001) {

            throw IllegalStateException(
                "Face alignment magnitude is invalid"
            )
        }

        val scale =
            magnitude /
                    denominator

        val cos =
            a / magnitude

        val sin =
            b / magnitude

        val translateX =
            targetCx -
                    scale *
                    (
                        cos * sourceCx -
                                sin * sourceCy
                        )

        val translateY =
            targetCy -
                    scale *
                    (
                        sin * sourceCx +
                                cos * sourceCy
                        )

        val matrix =
            Matrix()

        matrix.setValues(
            floatArrayOf(
                (scale * cos).toFloat(),
                (-scale * sin).toFloat(),
                translateX.toFloat(),

                (scale * sin).toFloat(),
                (scale * cos).toFloat(),
                translateY.toFloat(),

                0f,
                0f,
                1f
            )
        )

        return matrix
    }

    private fun createArcFaceEmbedding(
        face: Bitmap,
        session: OrtSession
    ): FloatArray {

        val aligned =
            Bitmap.createScaledBitmap(
                face,
                ARC_SIZE,
                ARC_SIZE,
                true
            )

        val pixels =
            IntArray(
                ARC_SIZE * ARC_SIZE
            )

        aligned.getPixels(
            pixels,
            0,
            ARC_SIZE,
            0,
            0,
            ARC_SIZE,
            ARC_SIZE
        )

        val buffer =
            FloatBuffer.allocate(
                3 *
                        ARC_SIZE *
                        ARC_SIZE
            )

        /*
         * ArcFace:
         * RGB
         * CHW
         * [-1, 1]
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
                    value / 127.5f - 1f
                )
            }
        }

        buffer.rewind()

        val tensor =
            OnnxTensor.createTensor(
                environment,
                buffer,
                longArrayOf(
                    1,
                    3,
                    ARC_SIZE.toLong(),
                    ARC_SIZE.toLong()
                )
            )

        val result =
            session.run(
                mapOf(
                    "input" to tensor
                )
            )

        val embedding =
            extractFloatArray(
                result[0].value
            )

        tensor.close()
        result.close()
        aligned.recycle()

        if (embedding.size != 512) {

            throw IllegalStateException(
                "ArcFace returned ${embedding.size} values"
            )
        }

        return normalizeEmbedding(
            embedding
        )
    }

    private fun runHyperSwap(
        embedding: FloatArray,
        targetFace: Bitmap,
        session: OrtSession
    ): HyperSwapResult {

        val target =
            Bitmap.createScaledBitmap(
                targetFace,
                HYPER_SIZE,
                HYPER_SIZE,
                true
            )

        val pixels =
            IntArray(
                HYPER_SIZE *
                        HYPER_SIZE
            )

        target.getPixels(
            pixels,
            0,
            HYPER_SIZE,
            0,
            0,
            HYPER_SIZE,
            HYPER_SIZE
        )

        val targetBuffer =
            FloatBuffer.allocate(
                3 *
                        HYPER_SIZE *
                        HYPER_SIZE
            )

        /*
         * HyperSwap:
         * RGB
         * CHW
         * mean = 0.5
         * std  = 0.5
         *
         * therefore:
         * value / 127.5 - 1
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

                targetBuffer.put(
                    value / 127.5f - 1f
                )
            }
        }

        targetBuffer.rewind()

        val sourceTensor =
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(
                    embedding
                ),
                longArrayOf(
                    1,
                    512
                )
            )

        val targetTensor =
            OnnxTensor.createTensor(
                environment,
                targetBuffer,
                longArrayOf(
                    1,
                    3,
                    HYPER_SIZE.toLong(),
                    HYPER_SIZE.toLong()
                )
            )

        val result =
            session.run(
                mapOf(
                    "source" to sourceTensor,
                    "target" to targetTensor
                )
            )

        val image =
            extractFloatArray(
                result[0].value
            )

        val mask =
            extractFloatArray(
                result[1].value
            )

        sourceTensor.close()
        targetTensor.close()
        result.close()
        target.recycle()

        val outputBitmap =
            imageToBitmap(
                image
            )

        return HyperSwapResult(
            bitmap = outputBitmap,
            mask = mask
        )
    }

    private fun pasteBack(
        original: Bitmap,
        swap: HyperSwapResult,
        transform: Matrix
    ): Bitmap {

        val output =
            original.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        val inverse =
            Matrix()

        if (!transform.invert(inverse)) {

            swap.bitmap.recycle()

            throw IllegalStateException(
                "Could not create paste transform"
            )
        }

        val warpedSwap =
            Bitmap.createBitmap(
                output.width,
                output.height,
                Bitmap.Config.ARGB_8888
            )

        val warpedMask =
            Bitmap.createBitmap(
                output.width,
                output.height,
                Bitmap.Config.ALPHA_8
            )

        val swapCanvas =
            Canvas(warpedSwap)

        val swapPaint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        swapCanvas.drawBitmap(
            swap.bitmap,
            inverse,
            swapPaint
        )

        val maskBitmap =
            maskToBitmap(
                swap.mask
            )

        val maskCanvas =
            Canvas(warpedMask)

        val maskPaint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        maskCanvas.drawBitmap(
            maskBitmap,
            inverse,
            maskPaint
        )

        val canvas =
            Canvas(output)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        paint.xfermode =
            PorterDuffXfermode(
                PorterDuff.Mode.DST_IN
            )

        val masked =
            Bitmap.createBitmap(
                output.width,
                output.height,
                Bitmap.Config.ARGB_8888
            )

        val maskedCanvas =
            Canvas(masked)

        maskedCanvas.drawBitmap(
            warpedSwap,
            0f,
            0f,
            null
        )

        maskedCanvas.drawBitmap(
            warpedMask,
            0f,
            0f,
            paint
        )

        paint.xfermode = null

        canvas.drawBitmap(
            masked,
            0f,
            0f,
            Paint(
                Paint.ANTI_ALIAS_FLAG
            )
        )

        warpedSwap.recycle()
        warpedMask.recycle()
        masked.recycle()
        maskBitmap.recycle()
        swap.bitmap.recycle()

        return output
    }

    private fun imageToBitmap(
        values: FloatArray
    ): Bitmap {

        val plane =
            HYPER_SIZE *
                    HYPER_SIZE

        val expected =
            plane * 3

        if (values.size < expected) {

            throw IllegalStateException(
                "Invalid HyperSwap image output"
            )
        }

        val pixels =
            IntArray(
                plane
            )

        for (i in pixels.indices) {

            val r =
                (
                    values[i] *
                            127.5f +
                            127.5f
                    )
                    .coerceIn(
                        0f,
                        255f
                    )
                    .toInt()

            val g =
                (
                    values[plane + i] *
                            127.5f +
                            127.5f
                    )
                    .coerceIn(
                        0f,
                        255f
                    )
                    .toInt()

            val b =
                (
                    values[plane * 2 + i] *
                            127.5f +
                            127.5f
                    )
                    .coerceIn(
                        0f,
                        255f
                    )
                    .toInt()

            pixels[i] =
                Color.rgb(
                    r,
                    g,
                    b
                )
        }

        val bitmap =
            Bitmap.createBitmap(
                HYPER_SIZE,
                HYPER_SIZE,
                Bitmap.Config.ARGB_8888
            )

        bitmap.setPixels(
            pixels,
            0,
            HYPER_SIZE,
            0,
            0,
            HYPER_SIZE,
            HYPER_SIZE
        )

        return bitmap
    }

    private fun maskToBitmap(
        values: FloatArray
    ): Bitmap {

        val expected =
            HYPER_SIZE *
                    HYPER_SIZE

        if (values.size < expected) {

            throw IllegalStateException(
                "Invalid HyperSwap mask output"
            )
        }

        val bitmap =
            Bitmap.createBitmap(
                HYPER_SIZE,
                HYPER_SIZE,
                Bitmap.Config.ALPHA_8
            )

        val buffer =
            java.nio.ByteBuffer.allocate(
                expected
            )

        for (i in 0 until expected) {

            buffer.put(
                (
                    values[i]
                        .coerceIn(
                            0f,
                            1f
                        ) *
                            255f
                    )
                    .toInt()
                    .toByte()
            )
        }

        buffer.rewind()

        bitmap.copyPixelsFromBuffer(
            buffer
        )

        return bitmap
    }

    private fun extractFloatArray(
        raw: Any
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

                val first =
                    raw.firstOrNull()

                when (first) {

                    is FloatArray ->
                        first

                    is DoubleArray ->
                        FloatArray(
                            first.size
                        ) {
                            first[it].toFloat()
                        }

                    is Array<*> -> {

                        first.flatMap { row ->

                            when (row) {

                                is FloatArray ->
                                    row.toList()

                                is DoubleArray ->
                                    row.map {
                                        it.toFloat()
                                    }

                                else ->
                                    emptyList()
                            }

                        }.toFloatArray()
                    }

                    else ->
                        throw IllegalStateException(
                            "Unexpected tensor output"
                        )
                }
            }

            else ->
                throw IllegalStateException(
                    "Unexpected tensor output"
                )
        }
    }

    private fun normalizeEmbedding(
        values: FloatArray
    ): FloatArray {

        var sum = 0.0

        for (value in values) {

            sum +=
                value.toDouble() *
                        value.toDouble()
        }

        val norm =
            sqrt(sum)
                .toFloat()

        if (norm < 0.000001f) {

            throw IllegalStateException(
                "Invalid ArcFace embedding"
            )
        }

        return FloatArray(
            values.size
        ) { index ->
            values[index] / norm
        }
    }

    private fun largestFace(
        faces: List<BlazeFaceResult>
    ): BlazeFaceResult {

        return faces.maxByOrNull {

            val width =
                max(
                    0f,
                    it.right - it.left
                )

            val height =
                max(
                    0f,
                    it.bottom - it.top
                )

            width * height

        } ?: throw IllegalStateException(
            "No usable face found"
        )
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

    private data class PreparedFace(
        val bitmap: Bitmap,
        val transform: Matrix
    )

    private data class HyperSwapResult(
        val bitmap: Bitmap,
        val mask: FloatArray
    )
}
