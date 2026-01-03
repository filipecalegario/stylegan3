import { IndividualCard } from './IndividualCard';
import './PopulationGrid.css';

export function PopulationGrid({
  population,
  fitness,
  getImageUrl,
  onFitnessChange,
  onAddFavorite,
  isLoading
}) {
  const gridSize = Math.ceil(Math.sqrt(population.length));

  return (
    <div
      className="population-grid"
      style={{
        gridTemplateColumns: `repeat(${gridSize}, 1fr)`
      }}
    >
      {population.map((individual) => (
        <IndividualCard
          key={individual.id}
          individual={individual}
          imageUrl={getImageUrl(individual)}
          fitness={fitness[individual.id] || 5.0}
          onFitnessChange={onFitnessChange}
          onAddFavorite={onAddFavorite}
          isLoading={isLoading}
        />
      ))}
    </div>
  );
}
