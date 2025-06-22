import React, { memo } from 'react';
import { SecureDOM } from '../utils/DOMSecurity';
import useStore from '../store/store';

const ZoneCard = memo(({ zone }) => {
  const { selectZone, currentZone } = useStore((state) => ({
    selectZone: state.selectZone,
    currentZone: state.currentZone
  }));
  const isActive = zone.id === currentZone;
  const hasAnomaly = zone.anomalies?.length > 0;
  const handleSelect = () => {
    selectZone(zone.id);
  };
  const getStatusClass = () => {
    if (zone.emergency_stop) return 'danger';
    if (zone.maintenance_mode) return 'warning';
    return 'online';
  };
  return (
    <div
      className={`zone-card metric-card ${isActive ? 'active' : ''} ${hasAnomaly ? 'anomaly-indicator' : ''}`}
      onClick={handleSelect}
      onKeyDown={(e) => e.key === 'Enter' && handleSelect()}
      tabIndex={0}
      role="button"
      aria-pressed={isActive}
    >
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-semibold">
            {SecureDOM.escapeHtml(zone.name)}
          </h3>
          <p className="text-sm text-gray-500">
            {SecureDOM.escapeHtml(zone.type || 'Production')}
          </p>
        </div>
        <div className="flex items-center">
          {hasAnomaly && (
            <i className="fas fa-exclamation-triangle text-yellow-500 mr-2" title="Anomaly detected" />
          )}
          <span className={`status-dot ${getStatusClass()}`} />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-gray-500">Units</p>
          <p className="font-semibold">{zone.units?.length || 0}</p>
        </div>
        <div>
          <p className="text-gray-500">Stage</p>
          <p className="font-semibold">
            {SecureDOM.escapeHtml(zone.crop_profile?.growth_stage || 'N/A')}
          </p>
        </div>
        <div>
          <p className="text-gray-500">Age</p>
          <p className="font-semibold">{zone.age_days || 0} days</p>
        </div>
        <div>
          <p className="text-gray-500">Yield</p>
          <p className="font-semibold text-green-600">
            {zone.total_yield?.toFixed(1) || '0.0'} kg
          </p>
        </div>
      </div>
      {zone.active_alarms > 0 && (
        <div className="mt-3 pt-3 border-t">
          <span className="text-xs text-red-600">
            <i className="fas fa-bell mr-1" />
            {zone.active_alarms} active alarm{zone.active_alarms > 1 ? 's' : ''}
          </span>
        </div>
      )}
    </div>
  );
});
ZoneCard.displayName = 'ZoneCard';
export default ZoneCard;
