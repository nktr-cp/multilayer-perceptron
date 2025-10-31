"use client"

import { Card } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Settings2 } from "lucide-react"

interface OptimizerPanelProps {
  optimizer: string
  onOptimizerChange: (value: string) => void
  learningRate: number
  onLearningRateChange: (value: number) => void
}

export function OptimizerPanel({
  optimizer,
  onOptimizerChange,
  learningRate,
  onLearningRateChange,
}: OptimizerPanelProps) {
  return (
    <Card className="glass-card p-6">
      <div className="flex items-center gap-2 mb-4">
        <Settings2 className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold text-foreground">Optimizer</h3>
      </div>

      <div className="space-y-4">
        {/* Optimizer Selection */}
        <div className="space-y-2">
          <Label className="text-sm text-muted-foreground">Algorithm</Label>
          <Select value={optimizer} onValueChange={onOptimizerChange}>
            <SelectTrigger className="bg-muted/50 border-border/50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="sgd">SGD</SelectItem>
              <SelectItem value="adam">Adam</SelectItem>
              <SelectItem value="rmsprop">RMSprop</SelectItem>
              <SelectItem value="adagrad">Adagrad</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Learning Rate */}
        <div className="space-y-2">
          <div className="flex justify-between">
            <Label className="text-sm text-muted-foreground">Learning Rate</Label>
            <span className="text-sm font-mono text-foreground">{learningRate.toFixed(4)}</span>
          </div>
          <Slider
            value={[learningRate * 1000]}
            onValueChange={([value]) => onLearningRateChange(value / 1000)}
            min={1}
            max={100}
            step={1}
            className="[&_[role=slider]]:bg-accent [&_[role=slider]]:border-accent"
          />
        </div>

        {/* Regularization */}
        <div className="space-y-2 pt-4 border-t border-border/50">
          <Label className="text-sm text-muted-foreground">L2 Regularization</Label>
          <Slider
            defaultValue={[1]}
            min={0}
            max={10}
            step={0.1}
            className="[&_[role=slider]]:bg-primary [&_[role=slider]]:border-primary"
          />
        </div>
      </div>
    </Card>
  )
}
