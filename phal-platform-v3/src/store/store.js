import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

const createZoneSlice = (set, get) => ({
  zones: new Map(),
  currentZone: null,
  setZones: (zones) => set((state) => {
    state.zones = new Map(zones.map(z => [z.id, z]));
  }),
  selectZone: (zoneId) => set((state) => {
    state.currentZone = zoneId;
  }),
  updateZone: (zoneId, updates) => set((state) => {
    const zone = state.zones.get(zoneId);
    if (zone) {
      state.zones.set(zoneId, { ...zone, ...updates });
    }
  }),
  getCurrentZone: () => {
    const state = get();
    return state.zones.get(state.currentZone);
  }
});

const createSensorSlice = (set) => ({
  sensorData: {},
  sensorHistory: {},
  updateSensorData: (zoneId, data) => set((state) => {
    state.sensorData[zoneId] = {
      ...state.sensorData[zoneId],
      ...data,
      lastUpdated: Date.now()
    };
  }),
  addSensorHistory: (zoneId, sensorType, value) => set((state) => {
    if (!state.sensorHistory[zoneId]) {
      state.sensorHistory[zoneId] = {};
    }
    if (!state.sensorHistory[zoneId][sensorType]) {
      state.sensorHistory[zoneId][sensorType] = [];
    }
    state.sensorHistory[zoneId][sensorType].push({
      value,
      timestamp: Date.now()
    });
    if (state.sensorHistory[zoneId][sensorType].length > 100) {
      state.sensorHistory[zoneId][sensorType].shift();
    }
  })
});

const createAlarmSlice = (set) => ({
  alarms: new Map(),
  alarmFilter: 'all',
  addAlarm: (alarm) => set((state) => {
    state.alarms.set(alarm.id, alarm);
  }),
  removeAlarm: (alarmId) => set((state) => {
    state.alarms.delete(alarmId);
  }),
  setAlarmFilter: (filter) => set((state) => {
    state.alarmFilter = filter;
  }),
  getFilteredAlarms: () => {
    const state = useStore.getState();
    const alarms = Array.from(state.alarms.values());
    if (state.alarmFilter === 'all') return alarms;
    return alarms.filter(alarm => alarm.severity === state.alarmFilter);
  }
});

const createUISlice = (set) => ({
  selectedTab: 'controls',
  modals: [],
  notifications: [],
  commandPaletteOpen: false,
  switchTab: (tab) => set((state) => {
    state.selectedTab = tab;
  }),
  showModal: (modal) => set((state) => {
    state.modals.push({ id: Date.now(), ...modal });
  }),
  closeModal: (modalId) => set((state) => {
    state.modals = state.modals.filter(m => m.id !== modalId);
  }),
  showNotification: (notification) => set((state) => {
    const id = Date.now();
    state.notifications.push({ id, ...notification });
    if (!notification.persistent) {
      setTimeout(() => {
        useStore.getState().removeNotification(id);
      }, 5000);
    }
  }),
  removeNotification: (id) => set((state) => {
    state.notifications = state.notifications.filter(n => n.id !== id);
  }),
  toggleCommandPalette: () => set((state) => {
    state.commandPaletteOpen = !state.commandPaletteOpen;
  })
});

const useStore = create(
  devtools(
    persist(
      subscribeWithSelector(
        immer((...args) => ({
          ...createZoneSlice(...args),
          ...createSensorSlice(...args),
          ...createAlarmSlice(...args),
          ...createUISlice(...args)
        }))
      ),
      {
        name: 'phal-storage',
        partialize: (state) => ({
          currentZone: state.currentZone,
          selectedTab: state.selectedTab,
          alarmFilter: state.alarmFilter
        })
      }
    ),
    { name: 'PHAL Store' }
  )
);

useStore.subscribe(
  (state) => state.sensorData,
  (sensorData) => {
    if (window.updateCharts) {
      window.updateCharts(sensorData);
    }
  }
);

export default useStore;
