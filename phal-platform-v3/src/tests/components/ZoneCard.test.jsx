import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ZoneCard from '@/components/ZoneCard';
import useStore from '@/store/store';

const mockZone = {
  id: 'zone-1',
  name: 'Test Zone',
  type: 'production',
  units: ['A1', 'A2'],
  age_days: 14,
  total_yield: 25.5,
  active_alarms: 2
};

describe('ZoneCard', () => {
  it('renders zone information correctly', () => {
    render(<ZoneCard zone={mockZone} />);
    expect(screen.getByText('Test Zone')).toBeInTheDocument();
    expect(screen.getByText('production')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('14 days')).toBeInTheDocument();
    expect(screen.getByText('25.5 kg')).toBeInTheDocument();
  });

  it('shows alarm indicator when active alarms exist', () => {
    render(<ZoneCard zone={mockZone} />);
    expect(screen.getByText(/2 active alarms/)).toBeInTheDocument();
  });

  it('handles zone selection', () => {
    const selectZone = vi.fn();
    useStore.setState({ selectZone });
    render(<ZoneCard zone={mockZone} />);
    fireEvent.click(screen.getByRole('button'));
    expect(selectZone).toHaveBeenCalledWith('zone-1');
  });

  it('supports keyboard navigation', () => {
    const selectZone = vi.fn();
    useStore.setState({ selectZone });
    render(<ZoneCard zone={mockZone} />);
    const card = screen.getByRole('button');
    fireEvent.keyDown(card, { key: 'Enter' });
    expect(selectZone).toHaveBeenCalledWith('zone-1');
  });
});
