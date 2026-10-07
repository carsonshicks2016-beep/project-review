using UnityEngine;
using Core.Physics;

namespace Core.Environment
{
    public class WeatherSystem : MonoBehaviour
    {
        public enum WeatherCondition { Clear, Rain, Snow, Night }
        
        [Header("Current Weather")]
        public WeatherCondition currentWeather = WeatherCondition.Clear;

        public event System.Action<WeatherCondition> OnWeatherChanged;

        [Header("Weather Modifiers")]
        public float rainGripMultiplier = 0.8f;
        public float snowGripMultiplier = 0.4f;

        [Header("Visual References")]
        public Light directionalLight;
        public Material skyboxMaterial;
        public GameObject rainParticles; // Should be camera-facing billboards for PS1 style
        public GameObject snowParticles;
        
        [Header("Night Mode Settings")]
        public float nightVisionTruncation = 0.5f; // Truncate LiDAR array by 50%

        private PacejkaTireModel[] allTires;

        void Start()
        {
            ApplyWeather();
        }

        public void SetWeather(WeatherCondition newWeather)
        {
            currentWeather = newWeather;
            ApplyWeather();
        }

        private void ApplyWeather()
        {
            allTires = FindObjectsByType<PacejkaTireModel>(FindObjectsSortMode.None);
            float baseGrip = 1.0f;
            
            // Turn off all effects initially
            if(rainParticles) rainParticles.SetActive(false);
            if(snowParticles) snowParticles.SetActive(false);
            if(directionalLight) directionalLight.intensity = 1.0f;

            switch (currentWeather)
            {
                case WeatherCondition.Clear:
                    baseGrip = 1.0f;
                    break;
                case WeatherCondition.Rain:
                    baseGrip = rainGripMultiplier;
                    if(rainParticles) rainParticles.SetActive(true);
                    if(directionalLight) directionalLight.intensity = 0.6f;
                    break;
                case WeatherCondition.Snow:
                    baseGrip = snowGripMultiplier;
                    if(snowParticles) snowParticles.SetActive(true);
                    if(directionalLight) directionalLight.intensity = 0.8f;
                    break;
                case WeatherCondition.Night:
                    baseGrip = 1.0f; // Grip is normal, but vision is truncated
                    if(directionalLight) directionalLight.intensity = 0.05f;
                    break;
            }

            // Apply grip globally to all active tires
            foreach (var tire in allTires)
            {
                // In a real system, this would modify the peak friction D in Pacejka
                // Here we apply it to a base multiplier
                tire.D = baseGrip;
            }

            OnWeatherChanged?.Invoke(currentWeather);
        }
    }
}
