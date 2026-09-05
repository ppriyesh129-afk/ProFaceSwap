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

private inline fun <T> Mat.use(block: (Mat) -> T): T {
    return try {
        block(this)
    } finally {
        release()
    }
}

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
         * STANDARD ARCFACE 112x112 TEMPLATE
         *
         * Used ONLY for the source identity embedding.
         */
        private val ARC_FACE_TEMPLATE = arrayOf(
            PointF(38.2946f, 51.6963f),
            PointF(73.5318f, 51.5014f),
            PointF(56.0252f, 71.7366f),
            PointF(41.5493f, 92.3655f),
            PointF(70.7299f, 92.2041f)
        )

        /*
         * HYPERSWAP arcface_128 TEMPLATE
         *
         * Used ONLY for the target image sent to HyperSwap.
         */
        private val HYPER_TEMPLATE = arrayOf(
            PointF(
                0.36167656f * HYPER_SIZE,
                0.40387734f * HYPER_SIZE
            ),
            PointF(
                0.63696719f * HYPER_SIZE,
                0.40235469f * HYPER_SIZE
            ),
            PointF(
                0.50019687f * HYPER_SIZE,
                0.56044219f * HYPER_SIZE
            ),
            PointF(
                0.38710391f * HYPER_SIZE,
                0.72160547f * HYPER_SIZE
            ),
            PointF(
                0.61507734f * HYPER_SIZE,
                0.72034453f * HYPER_SIZE
            )
        )
    }

    private val env: OrtEnvironment =
        OrtEnvironment.getEnvironment()

    private var faceDetector: BlazeFaceDetector? = null
    private var landmarkSession: OrtSession? = null
    private var arcFaceSession: OrtSession? = null
    private var hyperSwapSession: OrtSession? = null

    private var closed = false

    init {
        loadModels()
    }

    // ========================================================================
    // MODEL LOADING
    // ========================================================================

    private fun loadModels() {

        faceDetector =
            BlazeFaceDetector(context)

        landmarkSession =
            env.createSession(
                context.assets
                    .open("face_landmarker_Nx3x256x256.onnx")
                    .use { it.readBytes() },
                OrtSession.SessionOptions()
            )

        arcFaceSession =
            env.createSession(
                context.assets
                    .open("arcface_w600k_r50.onnx")
                    .use { it.readBytes() },
                OrtSession.SessionOptions()
            )

        hyperSwapSession =
            env.createSession(
                context.assets
                    .open("hyperswap_1a_256.onnx")
                    .use { it.readBytes() },
                OrtSession.SessionOptions()
            )
    }

    // ========================================================================
    // MAIN SWAP
    // ========================================================================

    fun processSwap(
        sourceBitmap: Bitmap,
        targetBitmap: Bitmap
    ): Bitmap {

        check(!closed) {
            "FaceSwapEngine is closed"
        }

        require(!sourceBitmap.isRecycled) {
            "Source bitmap is recycled"
        }

        require(!targetBitmap.isRecycled) {
            "Target bitmap is recycled"
        }

        /*
         * SOURCE:
         *
         * 478 landmarks
         *      ↓
         * ArcFace 112x112 alignment
         *      ↓
         * ArcFace 512-D embedding
         */
        val sourceEmbedding =
            createSourceArcFaceEmbedding(
                sourceBitmap
            )

        /*
         * TARGET:
         *
         * 478 landmarks
         *      ↓
         * HyperSwap arcface_128 256x256 alignment
         *      ↓
         * HyperSwap
         */
        val targetFace =
            prepareTargetFace(
                targetBitmap
            )

        val swappedBitmap =
            runHyperSwap(
                targetFace.alignedBitmap,
                sourceEmbedding
            )

        return pasteBack(
            targetBitmap,
            targetFace,
            swappedBitmap
        )
    }

    // ========================================================================
    // SOURCE ARC FACE EMBEDDING
    // ========================================================================

    private fun createSourceArcFaceEmbedding(
        sourceBitmap: Bitmap
    ): FloatArray {

        val sourceFace =
            detectAndPrepareFace(
                sourceBitmap
            )

        try {

            /*
             * CRITICAL:
             *
             * The source identity is aligned directly
             * to the STANDARD ArcFace 112 template.
             *
             * We do NOT:
             *
             * HyperSwap 256 → resize 112 → ArcFace
             */

            val arcAligned =
                warpBitmap(
                    sourceFace.faceBitmap,
                    sourceFace.localFivePoints,
                    ARC_FACE_TEMPLATE,
                    ARC_SIZE
                )

            try {

                return runArcFace(
                    arcAligned
                )

            } finally {

                if (!arcAligned.isRecycled) {
                    arcAligned.recycle()
                }
            }

        } finally {

            if (!sourceFace.faceBitmap.isRecycled) {
                sourceFace.faceBitmap.recycle()
            }
        }
    }

    // ========================================================================
    // TARGET FACE
    // ========================================================================

    private fun prepareTargetFace(
        targetBitmap: Bitmap
    ): PreparedFace {

        return detectAndPrepareFace(
            targetBitmap
        )
    }

    // ========================================================================
    // FACE DETECTION + LANDMARKS
    // ========================================================================

    private fun detectAndPrepareFace(
        bitmap: Bitmap
    ): PreparedFace {

        val detector =
            faceDetector
                ?: error(
                    "Face detector not initialized"
                )

        val landmarkSession =
            landmarkSession
                ?: error(
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

            val landmarks =
                detectLandmarks(
                    cropInfo.bitmap,
                    landmarkSession
                )

            /*
             * Convert landmark coordinates from
             * 256x256 model coordinates into the
             * detector crop's local pixel coordinates.
             */
            val localLandmarks =
                normalizeLandmarksToCrop(
                    landmarks,
                    cropInfo.bitmap.width,
                    cropInfo.bitmap.height
                )

            val localFivePoints =
                FaceAlignment.fivePoints(
                    localLandmarks
                )

            /*
             * Map those five points to ORIGINAL
             * target/source bitmap coordinates.
             */
            val originalFivePoints =
                Array(5) { i ->

                    PointF(
                        cropInfo.left +
                            localFivePoints[i].x,

                        cropInfo.top +
                            localFivePoints[i].y
                    )
                }

            /*
             * For target HyperSwap alignment we use
             * original image coordinates.
             */
            val hyperAligned =
                warpBitmap(
                    bitmap,
                    originalFivePoints,
                    HYPER_TEMPLATE,
                    HYPER_SIZE
                )

            /*
             * For source ArcFace alignment we need
             * the detector crop bitmap + LOCAL points.
             *
             * This gives a clean and stable 112x112
             * ArcFace input.
             */
            val faceBitmap =
                Bitmap.createBitmap(
                    cropInfo.bitmap
                )

            return PreparedFace(
                alignedBitmap = hyperAligned,
                faceBitmap = faceBitmap,
                localFivePoints = localFivePoints,
                originalFivePoints = originalFivePoints,
                originalLandmarks = localLandmarks
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
            detection.xMin * bitmap.width

        val top =
            detection.yMin * bitmap.height

        val right =
            detection.xMax * bitmap.width

        val bottom =
            detection.yMax * bitmap.height

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

            var indexR = 0
            var indexG = plane
            var indexB = plane * 2

            for (y in 0 until 256) {

                for (x in 0 until 256) {

                    val pixel =
                        resized.getPixel(
                            x,
                            y
                        )

                    input[indexR++] =
                        Color.red(pixel) *
                            INV_127_5 -
                            1f

                    input[indexG++] =
                        Color.green(pixel) *
                            INV_127_5 -
                            1f

                    input[indexB++] =
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
                        1,
                        3,
                        256,
                        256
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

                    return extract478Landmarks(
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

    private fun extract478Landmarks(
        output: Any
    ): Array<PointF> {

        when (output) {

            is FloatArray -> {

                if (
                    output.size >=
                    478 * 3
                ) {

                    return Array(478) { i ->

                        PointF(
                            output[i * 3],
                            output[i * 3 + 1]
                        )
                    }
                }
            }

            is Array<*> -> {

                /*
                 * [1][478][3]
                 */
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
        }

        throw IllegalStateException(
            "Unsupported 478-landmark output: " +
                output::class.java.name
        )
    }

    // ========================================================================
    // LANDMARK COORDINATES
    // ========================================================================

    private fun normalizeLandmarksToCrop(
        landmarks: Array<PointF>,
        cropWidth: Int,
        cropHeight: Int
    ): Array<PointF> {

        /*
         * FaceLandmarker outputs normalized coordinates
         * in the common [0,1] range.
         *
         * If the model happens to return 256-space
         * coordinates, convert those as well.
         */

        val maxX =
            landmarks.maxOf {
                it.x
            }

        val maxY =
            landmarks.maxOf {
                it.y
            }

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
                    p.x * cropWidth
                } else {
                    p.x * cropWidth / 256f
                }

            val y =
                if (normalized) {
                    p.y * cropHeight
                } else {
                    p.y * cropHeight / 256f
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

        require(sourcePoints.size >= 5)
        require(destinationPoints.size >= 5)

        val src =
            sourcePoints.take(5)

        val dst =
            destinationPoints.take(5)

        var srcCx = 0f
        var srcCy = 0f
        var dstCx = 0f
        var dstCy = 0f

        for (i in 0 until 5) {

            srcCx += src[i].x
            srcCy += src[i].y

            dstCx += dst[i].x
            dstCy += dst[i].y
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
                src[i].x - srcCx

            val sy =
                src[i].y - srcCy

            val dx =
                dst[i].x - dstCx

            val dy =
                dst[i].y - dstCy

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

        if (denom < 1e-8) {
            throw IllegalStateException(
                "Invalid face landmarks"
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
                ?: error(
                    "ArcFace model not initialized"
                )

        val input =
            FloatArray(
                3 *
                    ARC_SIZE *
                    ARC_SIZE
            )

        val plane =
            ARC_SIZE * ARC_SIZE

        var rIndex = 0
        var gIndex = plane
        var bIndex = plane * 2

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
                    1,
                    3,
                    ARC_SIZE,
                    ARC_SIZE
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
                    extractFloatArray(
                        result[0].value
                    )

                var norm =
                    0.0

                for (v in raw) {
                    norm +=
                        v.toDouble() *
                        v.toDouble()
                }

                norm =
                    sqrt(norm)

                if (norm < 1e-12) {
                    throw IllegalStateException(
                        "ArcFace returned invalid embedding"
                    )
                }

                return FloatArray(
                    raw.size
                ) { i ->
                    raw[i] /
                        norm.toFloat()
                }
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
                ?: error(
                    "HyperSwap model not initialized"
                )

        val plane =
            HYPER_SIZE *
                HYPER_SIZE

        val input =
            FloatArray(
                3 * plane
            )

        var rIndex = 0
        var gIndex = plane
        var bIndex = plane * 2

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
                    1,
                    3,
                    HYPER_SIZE,
                    HYPER_SIZE
                )
            )

        val embeddingTensor =
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(sourceEmbedding),
                longArrayOf(
                    1,
                    sourceEmbedding.size.toLong()
                )
            )

        try {

            val names =
                session.inputNames.toList()

            if (names.size < 2) {
                throw IllegalStateException(
                    "HyperSwap requires two inputs"
                )
            }

            val imageName =
                names.firstOrNull { name ->

                    val n =
                        name.lowercase()

                    n.contains("image") ||
                    n.contains("target") ||
                    n.contains("input")
                } ?: names[0]

            val embeddingName =
                names.firstOrNull { name ->

                    val n =
                        name.lowercase()

                    n.contains("source") ||
                    n.contains("embed") ||
                    n.contains("identity") ||
                    n.contains("latent")
                } ?: names[1]

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
            extractFloatArray(
                output
            )

        val plane =
            HYPER_SIZE *
                HYPER_SIZE

        require(
            data.size >=
                3 * plane
        ) {
            "Invalid HyperSwap output size"
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
                    y *
                        HYPER_SIZE +
                        x

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
            value *
                127.5f +
                127.5f
            )
            .coerceIn(
                0f,
                255f
            )
            .toInt()
    }

    // ========================================================================
    // GENERIC FLOAT OUTPUT
    // ========================================================================

    private fun extractFloatArray(
        output: Any
    ): FloatArray {

        if (output is FloatArray) {
            return output
        }

        if (output is Array<*>) {

            /*
             * [1][N]
             */
            if (
                output.size == 1 &&
                output[0] is FloatArray
            ) {
                return output[0] as FloatArray
            }
        }

        throw IllegalStateException(
            "Unsupported ONNX output: " +
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

        if (
            roiWidth > MAX_ROI_SIZE ||
            roiHeight > MAX_ROI_SIZE
        ) {

            return directPasteFallback(
                result,
                swappedBitmap,
                points
            )
        }

        val targetRoi =
            Bitmap.createBitmap(
                targetBitmap,
                roiLeft,
                roiTop,
                roiWidth,
                roiHeight
            )

        try {

            val swappedRoi =
                Bitmap.createBitmap(
                    roiWidth,
                    roiHeight,
                    Bitmap.Config.ARGB_8888
                )

            try {

                val canvas =
                    Canvas(swappedRoi)

                val matrix =
                    Matrix()

                matrix.setScale(
                    roiWidth.toFloat() /
                        HYPER_SIZE,

                    roiHeight.toFloat() /
                        HYPER_SIZE
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

            } finally {

                if (!swappedRoi.isRecycled) {
                    swappedRoi.recycle()
                }
            }

        } finally {

            if (!targetRoi.isRecycled) {
                targetRoi.recycle()
            }

            if (!swappedBitmap.isRecycled) {
                swappedBitmap.recycle()
            }
        }
    }

    // ========================================================================
    // OPENCV BLEND
    // ========================================================================

    private fun blendRoi(
        result: Bitmap,
        targetRoi: Bitmap,
        swappedRoi: Bitmap,
        left: Int,
        top: Int,
        landmarks: Array<PointF>
    ): Bitmap {

        try {

            targetRoi.use { targetMat ->

                swappedRoi.use { swappedMat ->

                    Mat().use { outputMat ->

                        Mat().use { mask ->

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

                            val centerX =
                                landmarks
                                    .map {
                                        it.x
                                    }
                                    .average() -
                                    left

                            val centerY =
                                landmarks
                                    .map {
                                        it.y
                                    }
                                    .average() -
                                    top

                            val minX =
                                landmarks
                                    .minOf {
                                        it.x
                                    }
                                    .toDouble() -
                                    left

                            val maxX =
                                landmarks
                                    .maxOf {
                                        it.x
                                    }
                                    .toDouble() -
                                    left

                            val minY =
                                landmarks
                                    .minOf {
                                        it.y
                                    }
                                    .toDouble() -
                                    top

                            val maxY =
                                landmarks
                                    .maxOf {
                                        it.y
                                    }
                                    .toDouble() -
                                    top

                            val rx =
                                max(
                                    1.0,
                                    (maxX - minX) *
                                        0.78
                                )

                            val ry =
                                max(
                                    1.0,
                                    (maxY - minY) *
                                        0.88
                                )

                            Imgproc.ellipse(
                                mask,
                                Point(
                                    centerX,
                                    centerY +
                                        ry * 0.04
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
                             * Feather the boundary.
                             */
                            var blur =
                                (
                                    min(
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

                            val center =
                                Point(
                                    centerX,
                                    centerY
                                )

                            try {

                                Photo.seamlessClone(
                                    swappedMat,
                                    targetMat,
                                    mask,
                                    center,
                                    outputMat,
                                    Photo.NORMAL_CLONE
                                )

                            } catch (
                                _: Throwable
                            ) {

                                /*
                                 * Safe alpha-copy fallback.
                                 */
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

                            if (!blended.isRecycled) {
                                blended.recycle()
                            }

                            return result
                        }
                    }
                }
            }

        } catch (
            _: Throwable
        ) {

            return directPasteFallback(
                result,
                swappedRoi,
                landmarks
            )
        }
    }

    // ========================================================================
    // SAFE FALLBACK
    // ========================================================================

    private fun directPasteFallback(
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

    private data class PreparedFace(
        val alignedBitmap: Bitmap,
        val faceBitmap: Bitmap,
        val localFivePoints: Array<PointF>,
        val originalFivePoints: Array<PointF>,
        val originalLandmarks: Array<PointF>
    )

    // ========================================================================
    // CLOSE
    // ========================================================================

    override fun close() {

        if (closed) {
            return
        }

        closed = true

        try {
            landmarkSession?.close()
        } catch (_: Throwable) {
        }

        try {
            arcFaceSession?.close()
        } catch (_: Throwable) {
        }

        try {
            hyperSwapSession?.close()
        } catch (_: Throwable) {
        }

        landmarkSession = null
        arcFaceSession = null
        hyperSwapSession = null
        faceDetector = null
    }
}
