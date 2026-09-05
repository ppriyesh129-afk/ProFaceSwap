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
import android.graphics.PointF
import java.nio.FloatBuffer
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo

class FaceSwapEngine(
    private val context: Context
) : AutoCloseable {

    companion object {

        private const val ARC_SIZE = 112
        private const val HYPER_SIZE = 256

        private const val MAX_ROI_SIZE = 768

        private const val ROI_PADDING_FACTOR = 0.18f
        private const val MIN_ROI_PADDING = 12f

        private const val INV_127_5 = 1f / 127.5f

        /*
         * Standard ArcFace 112x112 template.
         *
         * Used ONLY for source identity embedding.
         */
        private val ARC_FACE_TEMPLATE = arrayOf(
            PointF(38.2946f, 51.6963f),
            PointF(73.5318f, 51.5014f),
            PointF(56.0252f, 71.7366f),
            PointF(41.5493f, 92.3655f),
            PointF(70.7299f, 92.2041f)
        )

        /*
         * HyperSwap arcface_128 template.
         *
         * Used ONLY for target HyperSwap input.
         */
        private val HYPER_TEMPLATE = arrayOf(
            PointF(
                0.36167656f * HYPER_SIZE.toFloat(),
                0.40387734f * HYPER_SIZE.toFloat()
            ),
            PointF(
                0.63696719f * HYPER_SIZE.toFloat(),
                0.40235469f * HYPER_SIZE.toFloat()
            ),
            PointF(
                0.50019687f * HYPER_SIZE.toFloat(),
                0.56044219f * HYPER_SIZE.toFloat()
            ),
            PointF(
                0.38710391f * HYPER_SIZE.toFloat(),
                0.72160547f * HYPER_SIZE.toFloat()
            ),
            PointF(
                0.61507734f * HYPER_SIZE.toFloat(),
                0.72034453f * HYPER_SIZE.toFloat()
            )
        )
    }

    private val env: OrtEnvironment =
        OrtEnvironment.getEnvironment()

    private var faceDetector: BlazeFaceDetector? = null
    private var landmarkSession: OrtSession? = null
    private var arcFaceSession: OrtSession? = null
    private var hyperSwapSession: OrtSession? = null

    private var modelsLoaded = false
    private var closed = false

    init {
        loadModels()
    }

    // ========================================================================
    // MODEL LOADING
    // ========================================================================

    /*
     * Public Boolean API kept compatible with MainActivity.
     */
    fun loadModels(): Boolean {

        if (closed) {
            return false
        }

        if (modelsLoaded) {
            return true
        }

        return try {

            faceDetector =
                BlazeFaceDetector(context)

            landmarkSession =
                createSession(
                    "face_landmarker_Nx3x256x256.onnx"
                )

            arcFaceSession =
                createSession(
                    "arcface_w600k_r50.onnx"
                )

            hyperSwapSession =
                createSession(
                    "hyperswap_1a_256.onnx"
                )

            modelsLoaded = true

            true

        } catch (e: Exception) {

            modelsLoaded = false

            try {
                landmarkSession?.close()
            } catch (_: Exception) {
            }

            try {
                arcFaceSession?.close()
            } catch (_: Exception) {
            }

            try {
                hyperSwapSession?.close()
            } catch (_: Exception) {
            }

            landmarkSession = null
            arcFaceSession = null
            hyperSwapSession = null
            faceDetector = null

            false
        }
    }

    private fun createSession(
        assetName: String
    ): OrtSession {

        val bytes =
            context.assets
                .open(assetName)
                .use {
                    it.readBytes()
                }

        return env.createSession(
            bytes,
            OrtSession.SessionOptions()
        )
    }

    // ========================================================================
    // MAIN FACE SWAP
    // ========================================================================

    /*
     * Parameter names deliberately remain:
     *
     * source
     * target
     *
     * because MainActivity already calls:
     *
     * processSwap(
     *     source = ...,
     *     target = ...
     * )
     */
    fun processSwap(
        source: Bitmap,
        target: Bitmap
    ): Bitmap {

        check(!closed) {
            "FaceSwapEngine is closed"
        }

        if (!modelsLoaded) {
            if (!loadModels()) {
                throw IllegalStateException(
                    "Face swap models could not be loaded"
                )
            }
        }

        require(!source.isRecycled) {
            "Source bitmap is recycled"
        }

        require(!target.isRecycled) {
            "Target bitmap is recycled"
        }

        /*
         * SOURCE:
         *
         * face
         * -> 5 landmarks
         * -> STANDARD ArcFace 112x112
         * -> ArcFace
         * -> normalized 512-D embedding
         */
        val sourceEmbedding =
            createSourceEmbedding(source)

        /*
         * TARGET:
         *
         * face
         * -> 5 landmarks
         * -> HyperSwap arcface_128 256x256
         */
        val targetFace =
            prepareTargetFace(target)

        try {

            val swapped =
                runHyperSwap(
                    targetFace.alignedBitmap,
                    sourceEmbedding
                )

            return pasteBack(
                target,
                targetFace,
                swapped
            )

        } finally {

            if (!targetFace.alignedBitmap.isRecycled) {
                targetFace.alignedBitmap.recycle()
            }
        }
    }

    // ========================================================================
    // SOURCE EMBEDDING
    // ========================================================================

    private fun createSourceEmbedding(
        source: Bitmap
    ): FloatArray {

        val face =
            detectFaceAndLandmarks(source)

        try {

            /*
             * IMPORTANT CORRECTION:
             *
             * Source does NOT use HyperSwap 256 alignment.
             *
             * Source uses standard ArcFace 112 alignment.
             */
            val aligned =
                warpBitmap(
                    face.cropBitmap,
                    face.localFivePoints,
                    ARC_FACE_TEMPLATE,
                    ARC_SIZE
                )

            try {

                return runArcFace(aligned)

            } finally {

                if (!aligned.isRecycled) {
                    aligned.recycle()
                }
            }

        } finally {

            if (!face.cropBitmap.isRecycled) {
                face.cropBitmap.recycle()
            }
        }
    }

    // ========================================================================
    // TARGET PREPARATION
    // ========================================================================

    private fun prepareTargetFace(
        target: Bitmap
    ): PreparedFace {

        val face =
            detectFaceAndLandmarks(target)

        /*
         * Target uses HyperSwap's arcface_128 template.
         */
        val hyperAligned =
            warpBitmap(
                target,
                face.originalFivePoints,
                HYPER_TEMPLATE,
                HYPER_SIZE
            )

        return PreparedFace(
            alignedBitmap = hyperAligned,
            originalFivePoints = face.originalFivePoints
        )
    }

    // ========================================================================
    // FACE DETECTION
    // ========================================================================

    private fun detectFaceAndLandmarks(
        bitmap: Bitmap
    ): DetectedFace {

        val detector =
            faceDetector
                ?: throw IllegalStateException(
                    "Face detector not initialized"
                )

        val session =
            landmarkSession
                ?: throw IllegalStateException(
                    "Landmark model not initialized"
                )

        val detection =
            detector.detect(bitmap)
                ?: throw IllegalArgumentException(
                    "No face detected"
                )

        val cropInfo =
            createDetectorCrop(
                bitmap,
                detection
            )

        try {

            val rawLandmarks =
                detectLandmarks(
                    cropInfo.bitmap,
                    session
                )

            val localLandmarks =
                convertLandmarksToCrop(
                    rawLandmarks,
                    cropInfo.bitmap.width,
                    cropInfo.bitmap.height
                )

            val localFive =
                FaceAlignment.fivePoints(
                    localLandmarks
                )

            val originalFive =
                Array(5) { i ->

                    PointF(
                        cropInfo.left.toFloat() +
                            localFive[i].x,

                        cropInfo.top.toFloat() +
                            localFive[i].y
                    )
                }

            return DetectedFace(
                cropBitmap =
                    Bitmap.createBitmap(
                        cropInfo.bitmap
                    ),
                localFivePoints = localFive,
                originalFivePoints = originalFive
            )

        } finally {

            if (!cropInfo.bitmap.isRecycled) {
                cropInfo.bitmap.recycle()
            }
        }
    }

    // ========================================================================
    // DETECTOR CROP
    // ========================================================================

    private fun createDetectorCrop(
        bitmap: Bitmap,
        detection: BlazeFaceResult
    ): CropInfo {

        val left =
            detection.xMin *
                bitmap.width.toFloat()

        val top =
            detection.yMin *
                bitmap.height.toFloat()

        val right =
            detection.xMax *
                bitmap.width.toFloat()

        val bottom =
            detection.yMax *
                bitmap.height.toFloat()

        val width =
            max(
                1f,
                right - left
            )

        val height =
            max(
                1f,
                bottom - top
            )

        val padX =
            max(
                MIN_ROI_PADDING,
                width * ROI_PADDING_FACTOR
            )

        val padY =
            max(
                MIN_ROI_PADDING,
                height * ROI_PADDING_FACTOR
            )

        val cropLeft =
            max(
                0,
                (left - padX).toInt()
            )

        val cropTop =
            max(
                0,
                (top - padY).toInt()
            )

        val cropRight =
            minOf(
                bitmap.width,
                (right + padX).toInt()
            )

        val cropBottom =
            minOf(
                bitmap.height,
                (bottom + padY).toInt()
            )

        val cropWidth =
            max(
                1,
                cropRight - cropLeft
            )

        val cropHeight =
            max(
                1,
                cropBottom - cropTop
            )

        return CropInfo(
            bitmap =
                Bitmap.createBitmap(
                    bitmap,
                    cropLeft,
                    cropTop,
                    cropWidth,
                    cropHeight
                ),
            left = cropLeft,
            top = cropTop
        )
    }

    // ========================================================================
    // LANDMARK MODEL
    // ========================================================================

    private fun detectLandmarks(
        crop: Bitmap,
        session: OrtSession
    ): Array<PointF> {

        val resized =
            Bitmap.createScaledBitmap(
                crop,
                256,
                256,
                true
            )

        try {

            val plane =
                256 * 256

            val input =
                FloatArray(
                    3 * plane
                )

            var rIndex = 0
            var gIndex = plane
            var bIndex = 2 * plane

            for (y in 0 until 256) {

                for (x in 0 until 256) {

                    val pixel =
                        resized.getPixel(
                            x,
                            y
                        )

                    input[rIndex++] =
                        Color.red(pixel) *
                            INV_127_5 -
                            1f

                    input[gIndex++] =
                        Color.green(pixel) *
                            INV_127_5 -
                            1f

                    input[bIndex++] =
                        Color.blue(pixel) *
                            INV_127_5 -
                            1f
                }
            }

            val tensor =
                OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(input),
                    longArrayOf(
                        1L,
                        3L,
                        256L,
                        256L
                    )
                )

            try {

                val inputName =
                    session.inputNames
                        .iterator()
                        .next()

                session.run(
                    mapOf(
                        inputName to tensor
                    )
                ).use { result ->

                    return extractLandmarks(
                        result[0].value
                    )
                }

            } finally {

                tensor.close()
            }

        } finally {

            if (!resized.isRecycled) {
                resized.recycle()
            }
        }
    }

    // ========================================================================
    // LANDMARK OUTPUT
    // ========================================================================

    private fun extractLandmarks(
        output: Any
    ): Array<PointF> {

        if (output is FloatArray) {

            if (output.size >= 478 * 3) {

                return Array(478) { i ->

                    PointF(
                        output[i * 3],
                        output[i * 3 + 1]
                    )
                }
            }
        }

        if (output is Array<*>) {

            if (
                output.size == 1 &&
                output[0] is Array<*>
            ) {

                val level1 =
                    output[0] as Array<*>

                if (
                    level1.size >= 478 &&
                    level1[0] is FloatArray
                ) {

                    return Array(478) { i ->

                        val row =
                            level1[i]
                                as FloatArray

                        PointF(
                            row[0],
                            row[1]
                        )
                    }
                }
            }
        }

        throw IllegalStateException(
            "Unsupported face landmark output: " +
                output::class.java.name
        )
    }

    // ========================================================================
    // LANDMARK COORDINATES
    // ========================================================================

    private fun convertLandmarksToCrop(
        landmarks: Array<PointF>,
        cropWidth: Int,
        cropHeight: Int
    ): Array<PointF> {

        var maxX = 0f
        var maxY = 0f

        for (p in landmarks) {

            if (p.x > maxX) {
                maxX = p.x
            }

            if (p.y > maxY) {
                maxY = p.y
            }
        }

        /*
         * If output is normalized [0,1].
         */
        val normalized =
            maxX <= 1.5f &&
                maxY <= 1.5f

        return Array(
            landmarks.size
        ) { i ->

            val p =
                landmarks[i]

            val x =
                if (normalized) {
                    p.x *
                        cropWidth.toFloat()
                } else {
                    p.x *
                        cropWidth.toFloat() /
                        256f
                }

            val y =
                if (normalized) {
                    p.y *
                        cropHeight.toFloat()
                } else {
                    p.y *
                        cropHeight.toFloat() /
                        256f
                }

            PointF(
                x.coerceIn(
                    0f,
                    cropWidth.toFloat()
                ),
                y.coerceIn(
                    0f,
                    cropHeight.toFloat()
                )
            )
        }
    }

    // ========================================================================
    // SIMILARITY WARP
    // ========================================================================

    private fun warpBitmap(
        sourceBitmap: Bitmap,
        sourcePoints: Array<PointF>,
        destinationPoints: Array<PointF>,
        outputSize: Int
    ): Bitmap {

        require(sourcePoints.size >= 5) {
            "Need five source points"
        }

        require(destinationPoints.size >= 5) {
            "Need five destination points"
        }

        var srcCx = 0f
        var srcCy = 0f
        var dstCx = 0f
        var dstCy = 0f

        for (i in 0 until 5) {

            srcCx += sourcePoints[i].x
            srcCy += sourcePoints[i].y

            dstCx += destinationPoints[i].x
            dstCy += destinationPoints[i].y
        }

        srcCx /= 5f
        srcCy /= 5f

        dstCx /= 5f
        dstCy /= 5f

        var a = 0.0
        var b = 0.0
        var denom = 0.0

        for (i in 0 until 5) {

            val sx =
                sourcePoints[i].x -
                    srcCx

            val sy =
                sourcePoints[i].y -
                    srcCy

            val dx =
                destinationPoints[i].x -
                    dstCx

            val dy =
                destinationPoints[i].y -
                    dstCy

            a +=
                sx * dx +
                    sy * dy

            b +=
                sx * dy -
                    sy * dx

            denom +=
                sx * sx +
                    sy * sy
        }

        if (denom < 0.00000001) {
            throw IllegalStateException(
                "Invalid face landmark geometry"
            )
        }

        val scale =
            sqrt(
                a * a +
                    b * b
            ) / denom

        val angle =
            Math.atan2(
                b,
                a
            )

        val cosA =
            cos(angle)

        val sinA =
            sin(angle)

        val tx =
            dstCx -
                scale *
                (
                    cosA * srcCx -
                        sinA * srcCy
                )

        val ty =
            dstCy -
                scale *
                (
                    sinA * srcCx +
                        cosA * srcCy
                )

        val matrix =
            Matrix()

        matrix.setValues(
            floatArrayOf(
                (scale * cosA).toFloat(),
                (-scale * sinA).toFloat(),
                tx,

                (scale * sinA).toFloat(),
                (scale * cosA).toFloat(),
                ty,

                0f,
                0f,
                1f
            )
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
            sourceBitmap,
            matrix,
            paint
        )

        return output
    }

    // ========================================================================
    // ARCFACE
    // ========================================================================

    private fun runArcFace(
        bitmap: Bitmap
    ): FloatArray {

        val session =
            arcFaceSession
                ?: throw IllegalStateException(
                    "ArcFace model not initialized"
                )

        val plane =
            ARC_SIZE * ARC_SIZE

        val input =
            FloatArray(
                3 * plane
            )

        var rIndex = 0
        var gIndex = plane
        var bIndex = 2 * plane

        for (y in 0 until ARC_SIZE) {

            for (x in 0 until ARC_SIZE) {

                val pixel =
                    bitmap.getPixel(
                        x,
                        y
                    )

                input[rIndex++] =
                    Color.red(pixel) *
                        INV_127_5 -
                        1f

                input[gIndex++] =
                    Color.green(pixel) *
                        INV_127_5 -
                        1f

                input[bIndex++] =
                    Color.blue(pixel) *
                        INV_127_5 -
                        1f
            }
        }

        val tensor =
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(input),
                longArrayOf(
                    1L,
                    3L,
                    ARC_SIZE.toLong(),
                    ARC_SIZE.toLong()
                )
            )

        try {

            val inputName =
                session.inputNames
                    .iterator()
                    .next()

            session.run(
                mapOf(
                    inputName to tensor
                )
            ).use { result ->

                val raw =
                    extractFloatOutput(
                        result[0].value
                    )

                var sum = 0.0

                for (value in raw) {

                    sum +=
                        value.toDouble() *
                            value.toDouble()
                }

                val norm =
                    sqrt(sum)

                if (norm < 0.000000000001) {
                    throw IllegalStateException(
                        "Invalid ArcFace embedding"
                    )
                }

                val normalized =
                    FloatArray(
                        raw.size
                    )

                val normFloat =
                    norm.toFloat()

                for (i in raw.indices) {

                    normalized[i] =
                        raw[i] /
                            normFloat
                }

                return normalized
            }

        } finally {

            tensor.close()
        }
    }

    // ========================================================================
    // HYPERSWAP
    // ========================================================================

    private fun runHyperSwap(
        targetAligned: Bitmap,
        sourceEmbedding: FloatArray
    ): Bitmap {

        val session =
            hyperSwapSession
                ?: throw IllegalStateException(
                    "HyperSwap model not initialized"
                )

        val plane =
            HYPER_SIZE * HYPER_SIZE

        val input =
            FloatArray(
                3 * plane
            )

        var rIndex = 0
        var gIndex = plane
        var bIndex = 2 * plane

        for (y in 0 until HYPER_SIZE) {

            for (x in 0 until HYPER_SIZE) {

                val pixel =
                    targetAligned.getPixel(
                        x,
                        y
                    )

                input[rIndex++] =
                    Color.red(pixel) *
                        INV_127_5 -
                        1f

                input[gIndex++] =
                    Color.green(pixel) *
                        INV_127_5 -
                        1f

                input[bIndex++] =
                    Color.blue(pixel) *
                        INV_127_5 -
                        1f
            }
        }

        val imageTensor =
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(input),
                longArrayOf(
                    1L,
                    3L,
                    HYPER_SIZE.toLong(),
                    HYPER_SIZE.toLong()
                )
            )

        val embeddingTensor =
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(sourceEmbedding),
                longArrayOf(
                    1L,
                    sourceEmbedding.size.toLong()
                )
            )

        try {

            val names =
                session.inputNames.toList()

            if (names.size < 2) {
                throw IllegalStateException(
                    "HyperSwap model requires two inputs"
                )
            }

            var imageName =
                names[0]

            var embeddingName =
                names[1]

            for (name in names) {

                val lower =
                    name.lowercase()

                if (
                    lower.contains("source") ||
                    lower.contains("embed") ||
                    lower.contains("identity") ||
                    lower.contains("latent")
                ) {
                    embeddingName = name
                }

                if (
                    lower.contains("image") ||
                    lower.contains("target")
                ) {
                    imageName = name
                }
            }

            /*
             * Prevent accidentally selecting the same input.
             */
            if (imageName == embeddingName) {

                imageName = names[0]
                embeddingName = names[1]
            }

            session.run(
                mapOf(
                    imageName to imageTensor,
                    embeddingName to embeddingTensor
                )
            ).use { result ->

                return hyperOutputToBitmap(
                    result[0].value
                )
            }

        } finally {

            imageTensor.close()
            embeddingTensor.close()
        }
    }

    // ========================================================================
    // HYPERSWAP OUTPUT
    // ========================================================================

    private fun hyperOutputToBitmap(
        output: Any
    ): Bitmap {

        val data =
            extractFloatOutput(
                output
            )

        val plane =
            HYPER_SIZE * HYPER_SIZE

        if (data.size < 3 * plane) {
            throw IllegalStateException(
                "HyperSwap output has only " +
                    data.size +
                    " values"
            )
        }

        val bitmap =
            Bitmap.createBitmap(
                HYPER_SIZE,
                HYPER_SIZE,
                Bitmap.Config.ARGB_8888
            )

        for (y in 0 until HYPER_SIZE) {

            for (x in 0 until HYPER_SIZE) {

                val position =
                    y * HYPER_SIZE + x

                val r =
                    decodeHyperValue(
                        data[position]
                    )

                val g =
                    decodeHyperValue(
                        data[plane + position]
                    )

                val b =
                    decodeHyperValue(
                        data[2 * plane + position]
                    )

                bitmap.setPixel(
                    x,
                    y,
                    Color.rgb(
                        r,
                        g,
                        b
                    )
                )
            }
        }

        return bitmap
    }

    private fun decodeHyperValue(
        value: Float
    ): Int {

        return (
            value * 127.5f +
                127.5f
            )
            .coerceIn(
                0f,
                255f
            )
            .toInt()
    }

    // ========================================================================
    // OUTPUT ARRAY HELPER
    // ========================================================================

    private fun extractFloatOutput(
        output: Any
    ): FloatArray {

        if (output is FloatArray) {
            return output
        }

        if (output is Array<*>) {

            if (
                output.size == 1 &&
                output[0] is FloatArray
            ) {

                return output[0] as FloatArray
            }

            /*
             * Handle [1][1][N] style output.
             */
            if (
                output.size == 1 &&
                output[0] is Array<*>
            ) {

                val first =
                    output[0] as Array<*>

                if (
                    first.size == 1 &&
                    first[0] is FloatArray
                ) {

                    return first[0] as FloatArray
                }
            }
        }

        throw IllegalStateException(
            "Unsupported ONNX output type: " +
                output::class.java.name
        )
    }

    // ========================================================================
    // PASTE BACK
    // ========================================================================

    private fun pasteBack(
        targetBitmap: Bitmap,
        targetFace: PreparedFace,
        swappedBitmap: Bitmap
    ): Bitmap {

        val result =
            targetBitmap.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        val points =
            targetFace.originalFivePoints

        var minX =
            Float.MAX_VALUE

        var minY =
            Float.MAX_VALUE

        var maxX =
            Float.MIN_VALUE

        var maxY =
            Float.MIN_VALUE

        for (p in points) {

            minX =
                minOf(
                    minX,
                    p.x
                )

            minY =
                minOf(
                    minY,
                    p.y
                )

            maxX =
                maxOf(
                    maxX,
                    p.x
                )

            maxY =
                maxOf(
                    maxY,
                    p.y
                )
        }

        val faceWidth =
            maxX - minX

        val faceHeight =
            maxY - minY

        val padX =
            max(
                MIN_ROI_PADDING,
                faceWidth * 0.95f
            )

        val padY =
            max(
                MIN_ROI_PADDING,
                faceHeight * 0.85f
            )

        val roiLeft =
            max(
                0,
                (minX - padX).toInt()
            )

        val roiTop =
            max(
                0,
                (minY - padY).toInt()
            )

        val roiRight =
            minOf(
                targetBitmap.width,
                (maxX + padX).toInt()
            )

        val roiBottom =
            minOf(
                targetBitmap.height,
                (maxY + padY).toInt()
            )

        val roiWidth =
            max(
                1,
                roiRight - roiLeft
            )

        val roiHeight =
            max(
                1,
                roiBottom - roiTop
            )

        /*
         * Keep the crash protection from the previous version.
         */
        if (
            roiWidth > MAX_ROI_SIZE ||
            roiHeight > MAX_ROI_SIZE
        ) {

            return directFallback(
                result,
                swappedBitmap,
                points
            )
        }

        var targetRoi: Bitmap? = null
        var swappedRoi: Bitmap? = null

        try {

            targetRoi =
                Bitmap.createBitmap(
                    targetBitmap,
                    roiLeft,
                    roiTop,
                    roiWidth,
                    roiHeight
                )

            swappedRoi =
                Bitmap.createBitmap(
                    roiWidth,
                    roiHeight,
                    Bitmap.Config.ARGB_8888
                )

            val canvas =
                Canvas(swappedRoi)

            val matrix =
                Matrix()

            matrix.setScale(
                roiWidth.toFloat() /
                    HYPER_SIZE.toFloat(),

                roiHeight.toFloat() /
                    HYPER_SIZE.toFloat()
            )

            val paint =
                Paint(
                    Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
                )

            canvas.drawBitmap(
                swappedBitmap,
                matrix,
                paint
            )

            return blendRoi(
                result,
                targetRoi,
                swappedRoi,
                roiLeft,
                roiTop,
                points
            )

        } catch (_: Throwable) {

            return directFallback(
                result,
                swappedBitmap,
                points
            )

        } finally {

            if (
                targetRoi != null &&
                !targetRoi.isRecycled
            ) {
                targetRoi.recycle()
            }

            if (
                swappedRoi != null &&
                !swappedRoi.isRecycled
            ) {
                swappedRoi.recycle()
            }

            if (
                !swappedBitmap.isRecycled
            ) {
                swappedBitmap.recycle()
            }
        }
    }

    // ========================================================================
    // BLEND ROI
    // ========================================================================

    private fun blendRoi(
        result: Bitmap,
        targetRoi: Bitmap,
        swappedRoi: Bitmap,
        left: Int,
        top: Int,
        landmarks: Array<PointF>
    ): Bitmap {

        var targetMat: Mat? = null
        var swappedMat: Mat? = null
        var outputMat: Mat? = null
        var mask: Mat? = null

        try {

            targetMat =
                Mat()

            swappedMat =
                Mat()

            outputMat =
                Mat()

            mask =
                Mat()

            Utils.bitmapToMat(
                targetRoi,
                targetMat
            )

            Utils.bitmapToMat(
                swappedRoi,
                swappedMat
            )

            mask.create(
                targetMat.rows(),
                targetMat.cols(),
                CvType.CV_8UC1
            )

            mask.setTo(
                Scalar(0.0)
            )

            var centerX =
                0.0

            var centerY =
                0.0

            for (p in landmarks) {

                centerX +=
                    p.x.toDouble()

                centerY +=
                    p.y.toDouble()
            }

            centerX =
                centerX /
                    landmarks.size.toDouble() -
                    left.toDouble()

            centerY =
                centerY /
                    landmarks.size.toDouble() -
                    top.toDouble()

            var minX =
                Double.MAX_VALUE

            var minY =
                Double.MAX_VALUE

            var maxX =
                -Double.MAX_VALUE

            var maxY =
                -Double.MAX_VALUE

            for (p in landmarks) {

                val x =
                    p.x.toDouble() -
                        left.toDouble()

                val y =
                    p.y.toDouble() -
                        top.toDouble()

                minX =
                    minOf(
                        minX,
                        x
                    )

                minY =
                    minOf(
                        minY,
                        y
                    )

                maxX =
                    maxOf(
                        maxX,
                        x
                    )

                maxY =
                    maxOf(
                        maxY,
                        y
                    )
            }

            val rx =
                max(
                    1.0,
                    (maxX - minX) * 0.78
                )

            val ry =
                max(
                    1.0,
                    (maxY - minY) * 0.88
                )

            Imgproc.ellipse(
                mask,
                Point(
                    centerX,
                    centerY + ry * 0.04
                ),
                Size(
                    rx,
                    ry
                ),
                0.0,
                0.0,
                360.0,
                Scalar(255.0),
                -1
            )

            /*
             * Feather mask.
             */
            var blur =
                (
                    minOf(
                        targetMat.rows(),
                        targetMat.cols()
                    ) * 0.035
                ).toInt()

            if (blur < 9) {
                blur = 9
            }

            if (blur % 2 == 0) {
                blur++
            }

            Imgproc.GaussianBlur(
                mask,
                mask,
                Size(
                    blur.toDouble(),
                    blur.toDouble()
                ),
                0.0
            )

            val cloneCenter =
                Point(
                    centerX,
                    centerY
                )

            var cloned = false

            try {

                Photo.seamlessClone(
                    swappedMat,
                    targetMat,
                    mask,
                    cloneCenter,
                    outputMat,
                    Photo.NORMAL_CLONE
                )

                cloned = true

            } catch (_: Throwable) {
                cloned = false
            }

            if (!cloned) {

                targetMat.copyTo(
                    outputMat
                )

                swappedMat.copyTo(
                    outputMat,
                    mask
                )
            }

            val blended =
                Bitmap.createBitmap(
                    targetRoi.width,
                    targetRoi.height,
                    Bitmap.Config.ARGB_8888
                )

            try {

                Utils.matToBitmap(
                    outputMat,
                    blended
                )

                val canvas =
                    Canvas(result)

                val paint =
                    Paint(
                        Paint.ANTI_ALIAS_FLAG or
                            Paint.FILTER_BITMAP_FLAG
                    )

                canvas.drawBitmap(
                    blended,
                    left.toFloat(),
                    top.toFloat(),
                    paint
                )

                return result

            } finally {

                if (!blended.isRecycled) {
                    blended.recycle()
                }
            }

        } finally {

            targetMat?.release()
            swappedMat?.release()
            outputMat?.release()
            mask?.release()
        }
    }

    // ========================================================================
    // DIRECT FALLBACK
    // ========================================================================

    private fun directFallback(
        result: Bitmap,
        swappedBitmap: Bitmap,
        points: Array<PointF>
    ): Bitmap {

        val centerX =
            points
                .map {
                    it.x
                }
                .average()
                .toFloat()

        val centerY =
            points
                .map {
                    it.y
                }
                .average()
                .toFloat()

        val left =
            centerX -
                swappedBitmap.width / 2f

        val top =
            centerY -
                swappedBitmap.height / 2f

        val canvas =
            Canvas(result)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                    Paint.FILTER_BITMAP_FLAG
            )

        canvas.drawBitmap(
            swappedBitmap,
            left,
            top,
            paint
        )

        if (!swappedBitmap.isRecycled) {
            swappedBitmap.recycle()
        }

        return result
    }

    // ========================================================================
    // DATA CLASSES
    // ========================================================================

    private data class CropInfo(
        val bitmap: Bitmap,
        val left: Int,
        val top: Int
    )

    private data class DetectedFace(
        val cropBitmap: Bitmap,
        val localFivePoints: Array<PointF>,
        val originalFivePoints: Array<PointF>
    )

    private data class PreparedFace(
        val alignedBitmap: Bitmap,
        val originalFivePoints: Array<PointF>
    )

    // ========================================================================
    // CLOSE
    // ========================================================================

    override fun close() {

        if (closed) {
            return
        }

        closed = true
        modelsLoaded = false

        try {
            landmarkSession?.close()
        } catch (_: Exception) {
        }

        try {
            arcFaceSession?.close()
        } catch (_: Exception) {
        }

        try {
            hyperSwapSession?.close()
        } catch (_: Exception) {
        }

        landmarkSession = null
        arcFaceSession = null
        hyperSwapSession = null
        faceDetector = null
    }
}
