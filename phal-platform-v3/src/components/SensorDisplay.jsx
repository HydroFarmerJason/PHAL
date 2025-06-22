import React, { useEffect, useRef } from 'react';
import useStore from '../store/store';
import { formatSensorValue, getTrendIcon } from '../utils/formatters';

const SensorDisplay = ({ zoneId, sensorType, label, unit, targetValue }) => {
  const sensorData = useStore((state) => state.sensorData[zoneId]?.[sensorType]);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!sensorData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const history = useStore.getState().sensorHistory[zoneId]?.[sensorType] || [];
    if (history.length < 2) return;
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    const values = history.map(h => h.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((point, i) => {
      const x = (i / (history.length - 1)) * width;
      const y = height - ((point.value - min) / range) * height;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }, [sensorData, zoneId, sensorType]);

  if (!sensorData) {
    return (
      <div className="sensor-display skeleton">
        <div className="h-4 bg-gray-200 rounded w-20 mb-2"></div>
        <div className="h-8 bg-gray-200 rounded w-32"></div>
      </div>
    );
  }

  const { value, quality, trend } = sensorData;
  const formattedValue = formatSensorValue(value, sensorType);
  const isOutOfRange = targetValue && Math.abs(value - targetValue) > targetValue * 0.1;

  return (
    <div className="sensor-display">
      <div className="flex justify-between items-center">
        <div>
          <p className="text-sm text-gray-500">{label}</p>
          <p className={`text-2xl font-bold ${isOutOfRange ? 'text-yellow-600' : ''}`}>
            {formattedValue}{unit}
          </p>
          {targetValue && (
            <p className="text-xs text-gray-400">
              Target: {targetValue}{unit}
            </p>
          )}
        </div>
        <div className="text-right">
          <canvas ref={canvasRef} className="sparkline" width="100" height="30" />
          {trend !== undefined && (
            <div className={`trend-indicator ${getTrendIcon(trend).class}`}>
              <i className={`fas ${getTrendIcon(trend).icon}`} />
            </div>
          )}
        </div>
      </div>
      {quality < 0.8 && (
        <div className="mt-2 text-xs text-yellow-600">
          <i className="fas fa-exclamation-triangle mr-1" />
          Sensor quality: {(quality * 100).toFixed(0)}%
        </div>
      )}
    </div>
  );
};

export default SensorDisplay;
