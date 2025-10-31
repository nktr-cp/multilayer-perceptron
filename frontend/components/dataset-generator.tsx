"use client"

import { useState, useEffect, useCallback } from "react"
import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Sparkles, RefreshCw } from "lucide-react"

interface DataPoint {
  x: number
  y: number
  label: number
}

interface DatasetGeneratorProps {
  onDatasetGenerated?: (data: DataPoint[]) => void
}

export function DatasetGenerator({ onDatasetGenerated }: DatasetGeneratorProps) {
  const [datasetType, setDatasetType] = useState<"spiral" | "circular" | "xor">("spiral")
  const [sampleSize, setSampleSize] = useState(200)
  const [noiseLevel, setNoiseLevel] = useState(0.1)
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([])

  const generateSpiral = useCallback((samples: number, noise: number): DataPoint[] => {
    const points: DataPoint[] = []
    const n = Math.floor(samples / 2)

    for (let i = 0; i < n; i++) {
      const r = (i / n) * 5
      const t = (i / n) * 4 * Math.PI

      points.push({
        x: r * Math.cos(t) + (Math.random() - 0.5) * noise,
        y: r * Math.sin(t) + (Math.random() - 0.5) * noise,
        label: 0,
      })

      points.push({
        x: r * Math.cos(t + Math.PI) + (Math.random() - 0.5) * noise,
        y: r * Math.sin(t + Math.PI) + (Math.random() - 0.5) * noise,
        label: 1,
      })
    }

    return points
  }, [])

  const generateCircular = useCallback((samples: number, noise: number): DataPoint[] => {
    const points: DataPoint[] = []
    const n = Math.floor(samples / 2)

    for (let i = 0; i < n; i++) {
      const angle = Math.random() * 2 * Math.PI

      const r1 = 2 + (Math.random() - 0.5) * noise
      points.push({
        x: r1 * Math.cos(angle),
        y: r1 * Math.sin(angle),
        label: 0,
      })

      const r2 = 5 + (Math.random() - 0.5) * noise
      points.push({
        x: r2 * Math.cos(angle),
        y: r2 * Math.sin(angle),
        label: 1,
      })
    }

    return points
  }, [])

  const generateXOR = useCallback((samples: number, noise: number): DataPoint[] => {
    const points: DataPoint[] = []
    const n = Math.floor(samples / 4)

    for (let i = 0; i < n; i++) {
      points.push({
        x: Math.random() * 4 + 1 + (Math.random() - 0.5) * noise,
        y: Math.random() * 4 + 1 + (Math.random() - 0.5) * noise,
        label: 0,
      })

      points.push({
        x: Math.random() * 4 - 5 + (Math.random() - 0.5) * noise,
        y: Math.random() * 4 + 1 + (Math.random() - 0.5) * noise,
        label: 1,
      })

      points.push({
        x: Math.random() * 4 - 5 + (Math.random() - 0.5) * noise,
        y: Math.random() * 4 - 5 + (Math.random() - 0.5) * noise,
        label: 0,
      })

      points.push({
        x: Math.random() * 4 + 1 + (Math.random() - 0.5) * noise,
        y: Math.random() * 4 - 5 + (Math.random() - 0.5) * noise,
        label: 1,
      })
    }

    return points
  }, [])

  const generateDataset = useCallback(() => {
    let points: DataPoint[] = []

    switch (datasetType) {
      case "spiral":
        points = generateSpiral(sampleSize, noiseLevel)
        break
      case "circular":
        points = generateCircular(sampleSize, noiseLevel)
        break
      case "xor":
        points = generateXOR(sampleSize, noiseLevel)
        break
    }

    setDataPoints(points)
    if (onDatasetGenerated) {
      onDatasetGenerated(points)
    }
  }, [datasetType, sampleSize, noiseLevel, generateSpiral, generateCircular, generateXOR, onDatasetGenerated])

  useEffect(() => {
    generateDataset()
  }, [generateDataset])

  return (
    <Card className="glass-panel p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-primary" />
          <h3 className="font-semibold text-foreground">Dataset Generator</h3>
        </div>
        <Button size="sm" variant="ghost" onClick={generateDataset} className="h-8 w-8 p-0">
          <RefreshCw className="w-4 h-4" />
        </Button>
      </div>

      <div className="space-y-3">
        <Label className="text-sm text-muted-foreground">Pattern Type</Label>
        <div className="grid grid-cols-3 gap-2">
          {(["spiral", "circular", "xor"] as const).map((type) => (
            <Button
              key={type}
              variant={datasetType === type ? "default" : "outline"}
              size="sm"
              onClick={() => setDatasetType(type)}
              className="capitalize"
            >
              {type}
            </Button>
          ))}
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Label className="text-sm text-muted-foreground">Sample Size</Label>
          <span className="text-sm font-mono text-foreground">{sampleSize}</span>
        </div>
        <Slider
          value={[sampleSize]}
          onValueChange={([value]) => setSampleSize(value)}
          min={50}
          max={500}
          step={50}
          className="w-full"
        />
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Label className="text-sm text-muted-foreground">Noise Level</Label>
          <span className="text-sm font-mono text-foreground">{noiseLevel.toFixed(2)}</span>
        </div>
        <Slider
          value={[noiseLevel * 100]}
          onValueChange={([value]) => setNoiseLevel(value / 100)}
          min={0}
          max={50}
          step={5}
          className="w-full"
        />
      </div>

      <div className="relative aspect-square bg-background/50 rounded-lg border border-border/50 overflow-hidden">
        <svg viewBox="-6 -6 12 12" className="w-full h-full">
          <line x1="-6" y1="0" x2="6" y2="0" stroke="currentColor" strokeWidth="0.02" opacity="0.2" />
          <line x1="0" y1="-6" x2="0" y2="6" stroke="currentColor" strokeWidth="0.02" opacity="0.2" />

          {dataPoints.map((point, i) => (
            <motion.circle
              key={`${i}-${point.x}-${point.y}`}
              cx={point.x}
              cy={-point.y}
              r="0.08"
              fill={point.label === 0 ? "rgb(6, 182, 212)" : "rgb(236, 72, 153)"}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 0.8 }}
              transition={{ delay: i * 0.001, duration: 0.3 }}
            />
          ))}
        </svg>
      </div>

      <div className="text-xs text-muted-foreground text-center">{dataPoints.length} points generated</div>
    </Card>
  )
}
