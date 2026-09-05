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
import android.util.Log
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

class FaceSwapEngine(private val context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "FaceSwapEngine"
        
        private const val ARC_SIZE = 112
        private const val HYPER_SIZE = 256
        private const val MAX_ROI_SIZE = 768

        private const val ROI_PADDING_FACTOR = 0.18f
        private const val MIN_ROI_PADDING = 12f
        private const val INV_127_5 = 1f / 127.5f

        // Standard ArcFace 112x112 template (Source identity)
        private val ARC_FACE_TEMPLATE = arrayOf(
            PointF(38.2946f, 51.6963f),
            PointF(73.5318f, 51.5014f),
            PointF(56.0252f, 71.7366f),
            PointF(41.5493f, 92.3655f),
            PointF(70.7299f, 92.2041f)
        )

        // HyperSwap arcface_128 template (Target alignment)
        private val HYPER_TEMPLATE = arrayOf(
            PointF(0.36167656f * HYPER_SIZE, 0.40387734f * HYPER_SIZE),
            PointF(0.63696719f * HYPER_SIZE, 0.40235469f * HYPER_SIZE),
            PointF(0.50019687f * HYPER_SIZE, 0.56044219f * HYPER_SIZE),
            PointF(0.38710391f * HYPER_SIZE, 0.72160547f * HYPER_SIZE),
            PointF(0.61507734f * HYPER_SIZE, 0.72034453f * HYPER_SIZE)
        )
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

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

    fun loadModels(): Boolean {
        if (closed) return false
        if (modelsLoaded) return true

        return try {
            faceDetector = BlazeFaceDetector(context)
            landmarkSession = createSession("face_landmarker_Nx3x256x256.onnx")
            arcFaceSession = createSession("arcface_w600k_r50.onnx")
            hyperSwapSession = createSession("hyperswap_1a_256.onnx")
            modelsLoaded = true
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load models", e)
            modelsLoaded = false
            closeSessionsQuietly(landmarkSession, arcFaceSession, hyperSwapSession)
            landmarkSession = null
            arcFaceSession = null
            hyperSwapSession = null
            faceDetector = null
            false
        }
    }

    private fun createSession(assetName: String): OrtSession {
        val bytes = context.assets.open(assetName).use { it.readBytes() }
        return env.createSession(bytes, OrtSession.SessionOptions())
    }

    // ========================================================================
    // MAIN FACE SWAP
    // ========================================================================

    fun processSwap(source: Bitmap, target: Bitmap): Bitmap {
        check(!closed) { "FaceSwapEngine is closed" }
        if (!modelsLoaded && !loadModels()) {
            throw IllegalStateException("Face swap models could not be loaded")
        }
        require(!source.isRecycled) { "Source bitmap is recycled" }
        require(!target.isRecycled) { "Target bitmap is recycled" }

        val sourceEmbedding = createSourceEmbedding(source)
        val targetFace = prepareTargetFace(target)

        return targetFace.alignedBitmap.useBitmap { alignedBitmap ->
            runHyperSwap(alignedBitmap, sourceEmbedding).useBitmap { swapped ->
                pasteBack(target, targetFace, swapped)
            }
        }
    }

    // ========================================================================
    // SOURCE EMBEDDING
    // ========================================================================

    private fun createSourceEmbedding(source: Bitmap): FloatArray {
        val face = detectFaceAndLandmarks(source)
        return face.cropBitmap.useBitmap { crop ->
            warpBitmap(crop, face.localFivePoints, ARC_FACE_TEMPLATE, ARC_SIZE).useBitmap { aligned ->
                runArcFace(aligned)
            }
        }
    }

    // ========================================================================
    // TARGET PREPARATION
    // ========================================================================

    private fun prepareTargetFace(target: Bitmap): PreparedFace {
        val face = detectFaceAndLandmarks(target)
        val hyperAligned = warpBitmap(target, face.originalFivePoints, HYPER_TEMPLATE, HYPER_SIZE)
        return PreparedFace(hyperAligned, face.originalFivePoints)
    }

    // ========================================================================
    // FACE DETECTION
    // ========================================================================

    private fun detectFaceAndLandmarks(bitmap: Bitmap): DetectedFace {
        val detector = faceDetector ?: throw IllegalStateException("Face detector not initialized")
        val session = landmarkSession ?: throw IllegalStateException("Landmark model not initialized")

        val detection = detector.detect(bitmap) ?: throw IllegalArgumentException("No face detected")
        val cropInfo = createDetectorCrop(bitmap, detection)

        return cropInfo.bitmap.useBitmap { crop ->
            val rawLandmarks = detectLandmarks(crop, session)
            val localLandmarks = convertLandmarksToCrop(rawLandmarks, crop.width, crop.height)
            val localFive = FaceAlignment.fivePoints(localLandmarks)
            val originalFive = Array(5) { i ->
                PointF(cropInfo.left + localFive[i].x, cropInfo.top + localFive[i].y)
            }
            
            DetectedFace(Bitmap.createBitmap(crop), localFive, originalFive)
        }
    }

    // ========================================================================
    // DETECTOR CROP
    // ========================================================================

    private fun createDetectorCrop(bitmap: Bitmap, detection: BlazeFaceResult): CropInfo {
        val left = detection.xMin * bitmap.width
        val top = detection.yMin * bitmap.height
        val right = detection.xMax * bitmap.width
        val bottom = detection.yMax * bitmap.height

        val width = max(1f, right - left)
        val height = max(1f, bottom - top)

        val padX = max(MIN_ROI_PADDING, width * ROI_PADDING_FACTOR)
        val padY = max(MIN_ROI_PADDING, height * ROI_PADDING_FACTOR)

        val cropLeft = max(0, (left - padX).toInt())
        val cropTop = max(0, (top - padY).toInt())
        val cropRight = minOf(bitmap.width, (right + padX).toInt())
        val cropBottom = minOf(bitmap.height, (bottom + padY).toInt())

        val cropWidth = max(1, cropRight - cropLeft)
        val cropHeight = max(1, cropBottom - cropTop)

        val cropBmp = Bitmap.createBitmap(bitmap, cropLeft, cropTop, cropWidth, cropHeight)
        return CropInfo(cropBmp, cropLeft, cropTop)
    }

    // ========================================================================
    // LANDMARK MODEL
    // ========================================================================

    private fun detectLandmarks(crop: Bitmap, session: OrtSession): Array<PointF> {
        val resized = Bitmap.createScaledBitmap(crop, 256, 256, true)
        
        return try {
            val plane = 256 * 256
            val input = FloatArray(3 * plane)
            val pixels = IntArray(plane)
            resized.getPixels(pixels, 0, 256, 0, 0, 256, 256)

            var rIndex = 0
            var gIndex = plane
            var bIndex = 2 * plane

            for (pixel in pixels) {
                // High-speed bitwise extraction instead of Color.red/green/blue overhead
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF

                input[rIndex++] = r * INV_127_5 - 1f
                input[gIndex++] = g * INV_127_5 - 1f
                input[bIndex++] = b * INV_127_5 - 1f
            }

            OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1L, 3L, 256L, 256L)).use { tensor ->
                val inputName = session.inputNames.iterator().next()
                session.run(mapOf(inputName to tensor)).use { result ->
                    extractLandmarks(result[0].value)
                }
            }
        } finally {
            if (resized !== crop && !resized.isRecycled) {
                resized.recycle() // Clean up only if a new bitmap was actually created
            }
        }
    }

    private fun extractLandmarks(output: Any): Array<PointF> {
        if (output is FloatArray && output.size >= 478 * 3) {
            return Array(478) { i -> PointF(output[i * 3], output[i * 3 + 1]) }
        }
        
        if (output is Array<*> && output.size == 1 && output[0] is Array<*>) {
            val level1 = output[0] as Array<*>
            if (level1.size >= 478 && level1[0] is FloatArray) {
                return Array(478) { i ->
                    val row = level1[i] as FloatArray
                    PointF(row[0], row[1])
                }
            }
        }
        throw IllegalStateException("Unsupported face landmark output: " + output::class.java.name)
    }

    private fun convertLandmarksToCrop(landmarks: Array<PointF>, cropWidth: Int, cropHeight: Int): Array<PointF> {
        var maxX = 0f
        var maxY = 0f
        for (p in landmarks) {
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }

        val normalized = maxX <= 1.5f && maxY <= 1.5f

        return Array(landmarks.size) { i ->
            val p = landmarks[i]
            val x = if (normalized) p.x * cropWidth else p.x * cropWidth / 256f
            val y = if (normalized) p.y * cropHeight else p.y * cropHeight / 256f
            
            PointF(x.coerceIn(0f, cropWidth.toFloat()), y.coerceIn(0f, cropHeight.toFloat()))
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
        require(sourcePoints.size >= 5) { "Need five source points" }
        require(destinationPoints.size >= 5) { "Need five destination points" }

        var srcCx = 0f; var srcCy = 0f
        var dstCx = 0f; var dstCy = 0f

        for (i in 0 until 5) {
            srcCx += sourcePoints[i].x
            srcCy += sourcePoints[i].y
            dstCx += destinationPoints[i].x
            dstCy += destinationPoints[i].y
        }

        srcCx /= 5f; srcCy /= 5f
        dstCx /= 5f; dstCy /= 5f

        var a = 0.0; var b = 0.0
        var denom = 0.0

        for (i in 0 until 5) {
            val sx = sourcePoints[i].x - srcCx
            val sy = sourcePoints[i].y - srcCy
            val dx = destinationPoints[i].x - dstCx
            val dy = destinationPoints[i].y - dstCy
            
            a += sx * dx + sy * dy
            b += sx * dy - sy * dx
            denom += sx * sx + sy * sy
        }

        if (denom < 1e-8) throw IllegalStateException("Invalid face landmark geometry")

        val scale = sqrt(a * a + b * b) / denom
        val angle = Math.atan2(b, a)
        val cosA = cos(angle)
        val sinA = sin(angle)

        val tx = dstCx - scale * (cosA * srcCx - sinA * srcCy)
        val ty = dstCy - scale * (sinA * srcCx + cosA * srcCy)

        val matrix = Matrix().apply {
            setValues(floatArrayOf(
                (scale * cosA).toFloat(), (-scale * sinA).toFloat(), tx.toFloat(),
                (scale * sinA).toFloat(), (scale * cosA).toFloat(), ty.toFloat(),
                0f, 0f, 1f
            ))
        }

        val output = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output).apply { drawColor(Color.BLACK) }
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        
        canvas.drawBitmap(sourceBitmap, matrix, paint)
        return output
    }

    // ========================================================================
    // ARCFACE
    // ========================================================================

    private fun runArcFace(bitmap: Bitmap): FloatArray {
        val session = arcFaceSession ?: throw IllegalStateException("ArcFace model not initialized")

        val plane = ARC_SIZE * ARC_SIZE
        val input = FloatArray(3 * plane)
        val pixels = IntArray(plane)
        
        // Fast block pixel extraction
        bitmap.getPixels(pixels, 0, ARC_SIZE, 0, 0, ARC_SIZE, ARC_SIZE)

        var rIndex = 0; var gIndex = plane; var bIndex = 2 * plane
        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            input[rIndex++] = r * INV_127_5 - 1f
            input[gIndex++] = g * INV_127_5 - 1f
            input[bIndex++] = b * INV_127_5 - 1f
        }

        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1L, 3L, ARC_SIZE.toLong(), ARC_SIZE.toLong())).use { tensor ->
            val inputName = session.inputNames.iterator().next()
            session.run(mapOf(inputName to tensor)).use { result ->
                val raw = extractFloatOutput(result[0].value)
                val sumSq = raw.sumOf { it.toDouble() * it.toDouble() }
                val norm = sqrt(sumSq)

                if (norm < 1e-12) throw IllegalStateException("Invalid ArcFace embedding")
                
                val normFloat = norm.toFloat()
                return FloatArray(raw.size) { i -> raw[i] / normFloat }
            }
        }
    }

    // ========================================================================
    // HYPERSWAP
    // ========================================================================

    private fun runHyperSwap(targetAligned: Bitmap, sourceEmbedding: FloatArray): Bitmap {
        val session = hyperSwapSession ?: throw IllegalStateException("HyperSwap model not initialized")

        val plane = HYPER_SIZE * HYPER_SIZE
        val input = FloatArray(3 * plane)
        val pixels = IntArray(plane)
        
        // Fast block extraction
        targetAligned.getPixels(pixels, 0, HYPER_SIZE, 0, 0, HYPER_SIZE, HYPER_SIZE)

        var rIndex = 0; var gIndex = plane; var bIndex = 2 * plane
        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            input[rIndex++] = r * INV_127_5 - 1f
            input[gIndex++] = g * INV_127_5 - 1f
            input[bIndex++] = b * INV_127_5 - 1f
        }

        val imageTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1L, 3L, HYPER_SIZE.toLong(), HYPER_SIZE.toLong()))
        val embeddingTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(sourceEmbedding), longArrayOf(1L, sourceEmbedding.size.toLong()))

        return try {
            val names = session.inputNames.toList()
            check(names.size >= 2) { "HyperSwap model requires two inputs" }

            var imageName = names[0]
            var embeddingName = names[1]

            for (name in names) {
                val lower = name.lowercase()
                if (lower.contains("source") || lower.contains("embed") || lower.contains("identity") || lower.contains("latent")) {
                    embeddingName = name
                }
                if (lower.contains("image") || lower.contains("target")) {
                    imageName = name
                }
            }

            if (imageName == embeddingName) {
                imageName = names[0]; embeddingName = names[1]
            }

            session.run(mapOf(imageName to imageTensor, embeddingName to embeddingTensor)).use { result ->
                hyperOutputToBitmap(result[0].value)
            }
        } finally {
            imageTensor.close()
            embeddingTensor.close()
        }
    }

    private fun hyperOutputToBitmap(output: Any): Bitmap {
        val data = extractFloatOutput(output)
        val plane = HYPER_SIZE * HYPER_SIZE
        check(data.size >= 3 * plane) { "HyperSwap output has only ${data.size} values" }

        val pixels = IntArray(plane)
        for (i in 0 until plane) {
            val r = decodeHyperValue(data[i])
            val g = decodeHyperValue(data[plane + i])
            val b = decodeHyperValue(data[2 * plane + i])
            // Standard ARGB packed int format for fast bitmap building
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        
        return Bitmap.createBitmap(pixels, HYPER_SIZE, HYPER_SIZE, Bitmap.Config.ARGB_8888)
    }

    private fun decodeHyperValue(value: Float): Int {
        return (value * 127.5f + 127.5f).toInt().coerceIn(0, 255)
    }

    private fun extractFloatOutput(output: Any): FloatArray {
        return when (output) {
            is FloatArray -> output
            is Array<*> -> {
                if (output.size == 1 && output[0] is FloatArray) return output[0] as FloatArray
                if (output.size == 1 && output[0] is Array<*>) {
                    val first = output[0] as Array<*>
                    if (first.size == 1 && first[0] is FloatArray) return first[0] as FloatArray
                }
                throw IllegalStateException("Unsupported ONNX array layout")
            }
            else -> throw IllegalStateException("Unsupported ONNX output type: " + output::class.java.name)
        }
    }

    // ========================================================================
    // PASTE BACK
    // ========================================================================

    private fun pasteBack(targetBitmap: Bitmap, targetFace: PreparedFace, swappedBitmap: Bitmap): Bitmap {
        val result = targetBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val points = targetFace.originalFivePoints

        var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE; var maxY = Float.MIN_VALUE

        for (p in points) {
            if (p.x < minX) minX = p.x
            if (p.y < minY) minY = p.y
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }

        val faceWidth = maxX - minX
        val faceHeight = maxY - minY
        val padX = max(MIN_ROI_PADDING, faceWidth * 0.95f)
        val padY = max(MIN_ROI_PADDING, faceHeight * 0.85f)

        val roiLeft = max(0, (minX - padX).toInt())
        val roiTop = max(0, (minY - padY).toInt())
        val roiRight = minOf(targetBitmap.width, (maxX + padX).toInt())
        val roiBottom = minOf(targetBitmap.height, (maxY + padY).toInt())
        val roiWidth = max(1, roiRight - roiLeft)
        val roiHeight = max(1, roiBottom - roiTop)

        if (roiWidth > MAX_ROI_SIZE || roiHeight > MAX_ROI_SIZE) {
            return directFallback(result, swappedBitmap, points)
        }

        val targetRoi = Bitmap.createBitmap(targetBitmap, roiLeft, roiTop, roiWidth, roiHeight)
        val swappedRoi = Bitmap.createBitmap(roiWidth, roiHeight, Bitmap.Config.ARGB_8888)

        return try {
            val canvas = Canvas(swappedRoi)
            val matrix = Matrix().apply {
                setScale(roiWidth.toFloat() / HYPER_SIZE.toFloat(), roiHeight.toFloat() / HYPER_SIZE.toFloat())
            }
            val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
            canvas.drawBitmap(swappedBitmap, matrix, paint)

            blendRoi(result, targetRoi, swappedRoi, roiLeft, roiTop, points)
        } catch (e: Exception) {
            Log.e(TAG, "Exception during pasteBack", e)
            directFallback(result, swappedBitmap, points)
        } finally {
            if (!targetRoi.isRecycled) targetRoi.recycle()
            if (!swappedRoi.isRecycled) swappedRoi.recycle()
            // NOTE: swappedBitmap is strictly NOT recycled here anymore. ProcessSwap cleans it up.
        }
    }

    private fun blendRoi(
        result: Bitmap, targetRoi: Bitmap, swappedRoi: Bitmap, left: Int, top: Int, landmarks: Array<PointF>
    ): Bitmap {
        var targetMat: Mat? = null
        var swappedMat: Mat? = null
        var outputMat: Mat? = null
        var mask: Mat? = null

        try {
            targetMat = Mat()
            swappedMat = Mat()
            outputMat = Mat()
            mask = Mat()

            Utils.bitmapToMat(targetRoi, targetMat)
            Utils.bitmapToMat(swappedRoi, swappedMat)

            mask.create(targetMat.rows(), targetMat.cols(), CvType.CV_8UC1)
            mask.setTo(Scalar(0.0))

            var centerX = 0.0; var centerY = 0.0
            for (p in landmarks) {
                centerX += p.x
                centerY += p.y
            }
            centerX = centerX / landmarks.size - left
            centerY = centerY / landmarks.size - top

            var minX = Double.MAX_VALUE; var minY = Double.MAX_VALUE
            var maxX = -Double.MAX_VALUE; var maxY = -Double.MAX_VALUE

            for (p in landmarks) {
                val x = p.x.toDouble() - left
                val y = p.y.toDouble() - top
                if (x < minX) minX = x
                if (y < minY) minY = y
                if (x > maxX) maxX = x
                if (y > maxY) maxY = y
            }

            val rx = max(1.0, (maxX - minX) * 0.78)
            val ry = max(1.0, (maxY - minY) * 0.88)

            Imgproc.ellipse(mask, Point(centerX, centerY + ry * 0.04), Size(rx, ry), 0.0, 0.0, 360.0, Scalar(255.0), -1)

            var blur = (minOf(targetMat.rows(), targetMat.cols()) * 0.035).toInt()
            if (blur < 9) blur = 9
            if (blur % 2 == 0) blur++

            Imgproc.GaussianBlur(mask, mask, Size(blur.toDouble(), blur.toDouble()), 0.0)

            val cloneCenter = Point(centerX, centerY)
            var cloned = false

            try {
                Photo.seamlessClone(swappedMat, targetMat, mask, cloneCenter, outputMat, Photo.NORMAL_CLONE)
                cloned = true
            } catch (e: Throwable) {
                Log.e(TAG, "Seamless clone failed, dropping back to direct mask copy", e)
            }

            if (!cloned) {
                targetMat.copyTo(outputMat)
                swappedMat.copyTo(outputMat, mask)
            }

            val blended = Bitmap.createBitmap(targetRoi.width, targetRoi.height, Bitmap.Config.ARGB_8888)
            return blended.useBitmap { b ->
                Utils.matToBitmap(outputMat, b)
                val canvas = Canvas(result)
                val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
                canvas.drawBitmap(b, left.toFloat(), top.toFloat(), paint)
                result
            }
        } finally {
            targetMat?.release()
            swappedMat?.release()
            outputMat?.release()
            mask?.release()
        }
    }

    private fun directFallback(result: Bitmap, swappedBitmap: Bitmap, points: Array<PointF>): Bitmap {
        // Optimized to avoid creating temporary Lists just to find averages
        var sumX = 0f; var sumY = 0f
        for (p in points) {
            sumX += p.x
            sumY += p.y
        }
        val centerX = sumX / points.size
        val centerY = sumY / points.size

        val left = centerX - swappedBitmap.width / 2f
        val top = centerY - swappedBitmap.height / 2f

        val canvas = Canvas(result)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        canvas.drawBitmap(swappedBitmap, left, top, paint)

        return result
    }

    // ========================================================================
    // HELPER EXTENSIONS
    // ========================================================================

    /** Custom extension to safely execute a block with a Bitmap and recycle it securely. */
    private inline fun <T, R : Bitmap> R.useBitmap(block: (R) -> T): T {
        return try {
            block(this)
        } finally {
            if (!this.isRecycled) {
                this.recycle()
            }
        }
    }

    private fun closeSessionsQuietly(vararg sessions: OrtSession?) {
        for (session in sessions) {
            try { session?.close() } catch (e: Exception) { Log.e(TAG, "Failed closing session", e) }
        }
    }

    // ========================================================================
    // DATA CLASSES
    // ========================================================================

    private data class CropInfo(val bitmap: Bitmap, val left: Int, val top: Int)
    
    private data class DetectedFace(
        val cropBitmap: Bitmap, val localFivePoints: Array<PointF>, val originalFivePoints: Array<PointF>
    )
    
    private data class PreparedFace(val alignedBitmap: Bitmap, val originalFivePoints: Array<PointF>)

    // ========================================================================
    // CLOSE
    // ========================================================================

    override fun close() {
        if (closed) return
        closed = true
        modelsLoaded = false

        closeSessionsQuietly(landmarkSession, arcFaceSession, hyperSwapSession)
        
        landmarkSession = null
        arcFaceSession = null
        hyperSwapSession = null
        faceDetector = null
    }
}
