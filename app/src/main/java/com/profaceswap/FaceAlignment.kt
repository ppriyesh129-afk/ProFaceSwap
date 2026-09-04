package com.profaceswap

import android.graphics.Bitmap
import android.graphics.Matrix
import kotlin.math.sqrt

object FaceAlignment {

    data class AlignedFace(
        val bitmap: Bitmap,
        val matrix: Matrix
    )

    private val ARC_FACE_112_V2 = arrayOf(
        floatArrayOf(0.34191607f * 112f, 0.46157411f * 112f),
        floatArrayOf(0.65653393f * 112f, 0.45983393f * 112f),
        floatArrayOf(0.50022500f * 112f, 0.64050536f * 112f),
        floatArrayOf(0.37097589f * 112f, 0.82469196f * 112f),
        floatArrayOf(0.63151696f * 112f, 0.82325089f * 112f)
    )

    private val FFHQ_512 = arrayOf(
        floatArrayOf(0.37691676f * 256f, 0.46864664f * 256f),
        floatArrayOf(0.62285697f * 256f, 0.46912813f * 256f),
        floatArrayOf(0.50123859f * 256f, 0.61331904f * 256f),
        floatArrayOf(0.39308822f * 256f, 0.72541100f * 256f),
        floatArrayOf(0.61150205f * 256f, 0.72490465f * 256f)
    )

    fun align(
        bitmap: Bitmap,
        landmarks: Array<FloatArray>,
        outputSize: Int,
        useArcFaceTemplate: Boolean
    ): AlignedFace? {

        if (landmarks.size < 292) {
            return null
        }

        val source = fivePoints(landmarks)

        val template = if (useArcFaceTemplate) {
            ARC_FACE_112_V2
        } else {
            FFHQ_512.map { point ->
                floatArrayOf(
                    point[0] * outputSize / 256f,
                    point[1] * outputSize / 256f
                )
            }.toTypedArray()
        }

        val matrix = similarityTransform(
            source,
            template
        ) ?: return null

        val output =
            Bitmap.createBitmap(
                outputSize,
                outputSize,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            android.graphics.Canvas(output)

        canvas.drawColor(android.graphics.Color.BLACK)

        val paint =
            android.graphics.Paint(
                android.graphics.Paint.ANTI_ALIAS_FLAG or
                        android.graphics.Paint.FILTER_BITMAP_FLAG
            )

        canvas.drawBitmap(
            bitmap,
            matrix,
            paint
        )

        return AlignedFace(
            bitmap = output,
            matrix = matrix
        )
    }

    fun fivePoints(
        landmarks: Array<FloatArray>
    ): Array<FloatArray> {

        return arrayOf(
            point(landmarks, 33),
            point(landmarks, 263),
            point(landmarks, 1),
            point(landmarks, 61),
            point(landmarks, 291)
        )
    }

    private fun point(
        landmarks: Array<FloatArray>,
        index: Int
    ): FloatArray {

        return floatArrayOf(
            landmarks[index][0],
            landmarks[index][1]
        )
    }

    private fun similarityTransform(
        source: Array<FloatArray>,
        target: Array<FloatArray>
    ): Matrix? {

        if (source.size != 5 || target.size != 5) {
            return null
        }

        var sourceCx = 0f
        var sourceCy = 0f
        var targetCx = 0f
        var targetCy = 0f

        for (i in 0 until 5) {
            sourceCx += source[i][0]
            sourceCy += source[i][1]
            targetCx += target[i][0]
            targetCy += target[i][1]
        }

        sourceCx /= 5f
        sourceCy /= 5f
        targetCx /= 5f
        targetCy /= 5f

        var a = 0.0
        var b = 0.0
        var denominator = 0.0

        for (i in 0 until 5) {

            val sx =
                source[i][0] - sourceCx

            val sy =
                source[i][1] - sourceCy

            val tx =
                target[i][0] - targetCx

            val ty =
                target[i][1] - targetCy

            a += sx * tx + sy * ty
            b += sx * ty - sy * tx

            denominator +=
                sx.toDouble() * sx +
                        sy.toDouble() * sy
        }

        if (denominator < 0.000001) {
            return null
        }

        val scale =
            sqrt(a * a + b * b) / denominator

        val cos =
            a / sqrt(a * a + b * b)

        val sin =
            b / sqrt(a * a + b * b)

        val translateX =
            targetCx -
                    scale *
                    (cos * sourceCx - sin * sourceCy)

        val translateY =
            targetCy -
                    scale *
                    (sin * sourceCx + cos * sourceCy)

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
}
