#pragma once

#include <string>
#include <vector>

namespace bare_nn::charts {

struct ChartTheme {
    std::string background = "#eef2f3";
    std::string panel = "#ffffff";
    std::string ink = "#2f4a57";
    std::string muted = "#8aa0aa";
    std::string grid = "#cfd8dd";
    std::vector<std::string> palette = {
        "#385663", "#4f6f7b", "#6f8f9b", "#8faab4", "#2d3f49", "#6d7f87"
    };
};

struct ChartConfig {
    int width = 920;
    int height = 560;
    int padding = 64;
    ChartTheme theme{};
    std::string title;
};

bool create_bar_chart(
    const std::string& output_svg,
    const std::vector<std::string>& labels,
    const std::vector<double>& values,
    const ChartConfig& config = {});

bool create_pie_chart(
    const std::string& output_svg,
    const std::vector<std::string>& labels,
    const std::vector<double>& values,
    const ChartConfig& config = {});

bool create_line_chart(
    const std::string& output_svg,
    const std::vector<std::string>& x_labels,
    const std::vector<double>& values,
    const ChartConfig& config = {});

bool create_area_line_chart(
    const std::string& output_svg,
    const std::vector<std::string>& x_labels,
    const std::vector<double>& values,
    const ChartConfig& config = {});

bool create_multi_line_chart(
    const std::string& output_svg,
    const std::vector<std::string>& x_labels,
    const std::vector<std::vector<double>>& series,
    const std::vector<std::string>& series_names,
    const ChartConfig& config = {});

bool create_grouped_bar_chart(
    const std::string& output_svg,
    const std::vector<std::string>& labels,
    const std::vector<double>& series_a,
    const std::vector<double>& series_b,
    const ChartConfig& config = {});

bool create_horizontal_progress_chart(
    const std::string& output_svg,
    const std::vector<std::string>& labels,
    const std::vector<double>& values,
    const ChartConfig& config = {});

bool create_horizontal_ranked_bar_chart(
    const std::string& output_svg,
    const std::vector<std::string>& labels,
    const std::vector<double>& values,
    const ChartConfig& config = {});

bool create_stacked_bar_chart(
    const std::string& output_svg,
    const std::vector<std::string>& categories,
    const std::vector<std::string>& stack_labels,
    const std::vector<std::vector<double>>& stacks,
    const ChartConfig& config = {});

bool create_theme_showcase(
    const std::string& output_svg,
    const ChartConfig& config = {});

}  // namespace bare_nn::charts
