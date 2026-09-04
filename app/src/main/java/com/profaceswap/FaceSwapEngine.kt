package com.profaceswap

import android.content.Context
import android.graphics.Bitmap

class FaceSwapEngine(private val context: Context) {

    fun loadModels(): Boolean {
        return true
    }

    fun processSwap(
        target: Bitmap,
        source: Bitmap
    ): Bitmap {
        return target
    }
}
