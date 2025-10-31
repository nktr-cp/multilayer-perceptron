"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Play, Pause, RotateCcw } from "lucide-react"
import { Line } from "react-chartjs-2"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js"

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

interface TrainingDashboardProps {
  isTraining: boolean
  onToggleTraining: () => void
}

export function TrainingDashboard({ isTraining, onToggleTraining }: TrainingDashboardProps) {
  const [epoch, setEpoch] = useState(0)
  const [loss, setLoss] = useState(2.5)
  const [accuracy, setAccuracy] = useState(0.33)
  const [lossHistory, setLossHistory] = useState<number[]>([2.5])
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([0.33])

  useEffect(() => {
    if (!isTraining) return

    const interval = setInterval(() => {
      setEpoch((prev) => prev + 1)
      setLoss((prev) => Math.max(0.01, prev * 0.95 + (Math.random() - 0.5) * 0.1))
      setAccuracy((prev) => Math.min(0.99, prev + Math.random() * 0.02))

      setLossHistory((prev) => [...prev.slice(-49), loss])
      setAccuracyHistory((prev) => [...prev.slice(-49), accuracy])
    }, 500)

    return () => clearInterval(interval)
  }, [isTraining, loss, accuracy])

  const handleReset = () => {
    setEpoch(0)
    setLoss(2.5)
    setAccuracy(0.33)
    setLossHistory([2.5])
    setAccuracyHistory([0.33])
  }

  const chartData = {
    labels: Array.from({ length: lossHistory.length }, (_, i) => i.toString()),
    datasets: [
      {
        label: "Loss",
        data: lossHistory,
        borderColor: "oklch(0.7 0.25 195)",
        backgroundColor: "oklch(0.7 0.25 195 / 0.1)",
        fill: true,
        tension: 0.4,
        pointRadius: 0,
      },
    ],
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        grid: {
          color: "oklch(0.25 0.03 265 / 0.2)",
        },
        ticks: {
          color: "oklch(0.65 0.02 265)",
        },
      },
    },
  }

  return (
    <Card className="glass-card glow-magenta p-6">
      <h3 className="text-lg font-semibold text-foreground mb-4">Training Metrics</h3>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">Epoch</p>
          <p className="text-2xl font-bold text-foreground">{epoch}</p>
        </div>
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">Loss</p>
          <p className="text-2xl font-bold text-primary">{loss.toFixed(4)}</p>
        </div>
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">Accuracy</p>
          <p className="text-2xl font-bold text-accent">{(accuracy * 100).toFixed(2)}%</p>
        </div>
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">Speed</p>
          <p className="text-2xl font-bold text-secondary">2.1 ep/s</p>
        </div>
      </div>

      {/* Loss Chart */}
      <div className="h-32 mb-6">
        <Line data={chartData} options={chartOptions} />
      </div>

      {/* Controls */}
      <div className="flex gap-2">
        <Button onClick={onToggleTraining} className="flex-1 bg-primary hover:bg-primary/90 text-primary-foreground">
          {isTraining ? (
            <>
              <Pause className="w-4 h-4 mr-2" />
              Pause
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Train
            </>
          )}
        </Button>
        <Button onClick={handleReset} variant="outline" className="border-border/50 hover:bg-muted/50 bg-transparent">
          <RotateCcw className="w-4 h-4" />
        </Button>
      </div>
    </Card>
  )
}
