export function formatSensorValue(value, type) {
  if (value === undefined || value === null) return 'N/A';
  switch (type) {
    case 'temperature':
      return value.toFixed(1);
    case 'humidity':
    case 'co2':
    case 'ph':
    case 'ec':
      return value.toFixed(2);
    default:
      return String(value);
  }
}

export function getTrendIcon(trend) {
  if (trend > 0) return { icon: 'fa-arrow-up', class: 'trend-up' };
  if (trend < 0) return { icon: 'fa-arrow-down', class: 'trend-down' };
  return { icon: 'fa-minus', class: 'trend-flat' };
}
