#include <iostream>
#include <vector>

#include "charts_api.h"

int main() {
    using namespace bare_nn::charts;

    ChartConfig cfg;
    cfg.width = 980;
    cfg.height = 620;
    cfg.title = "Operations Dashboard";

    const std::vector<std::string> months = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN"};
    const std::vector<double> sales = {62, 88, 54, 79, 68, 92};

    if (!create_bar_chart("examples/charts/bar_chart.svg", months, sales, cfg)) {
        std::cerr << "bar chart failed\n";
        return 1;
    }

    const std::vector<std::string> pie_labels = {"Group 1", "Group 2", "Group 3", "Group 4"};
    const std::vector<double> pie_values = {25, 42, 18, 50};
    if (!create_pie_chart("examples/charts/pie_chart.svg", pie_labels, pie_values, cfg)) {
        std::cerr << "pie chart failed\n";
        return 1;
    }

    const std::vector<std::string> x_labels = {"10", "20", "30", "40", "50", "60"};
    const std::vector<double> trend = {38, 42, 31, 50, 47, 58};
    if (!create_line_chart("examples/charts/line_chart.svg", x_labels, trend, cfg)) {
        std::cerr << "line chart failed\n";
        return 1;
    }

    const std::vector<std::string> categories = {"Q1", "Q2", "Q3", "Q4"};
    const std::vector<std::string> stack_labels = {"Ops", "Sales", "R&D"};
    const std::vector<std::vector<double>> stacks = {
        {20, 28, 24, 30},
        {18, 16, 22, 19},
        {14, 19, 17, 20}
    };
    if (!create_stacked_bar_chart(
            "examples/charts/stacked_bar_chart.svg", categories, stack_labels, stacks, cfg)) {
        std::cerr << "stacked bar chart failed\n";
        return 1;
    }

    ChartConfig area_cfg = cfg;
    area_cfg.height = 360;
    area_cfg.title = "Striped Area Trend";
    const std::vector<std::string> area_x = {"A", "B", "C", "D", "E", "F", "G", "H"};
    const std::vector<double> area_y = {42, 37, 45, 54, 51, 57, 66, 74};
    if (!create_area_line_chart("examples/charts/area_line_chart.svg", area_x, area_y, area_cfg)) {
        std::cerr << "area line chart failed\n";
        return 1;
    }

    ChartConfig multi_cfg = cfg;
    multi_cfg.title = "Dual-Line Panel";
    const std::vector<std::string> multi_x = {"A", "B", "C", "D", "E", "F", "G", "H", "A"};
    const std::vector<std::vector<double>> multi_series = {
        {36, 34, 40, 58, 52, 89, 46, 68, 75},
        {14, 22, 30, 54, 38, 28, 50, 44, 18}
    };
    const std::vector<std::string> multi_names = {"Series 1", "Series 2"};
    if (!create_multi_line_chart(
            "examples/charts/multi_line_chart.svg", multi_x, multi_series, multi_names, multi_cfg)) {
        std::cerr << "multi line chart failed\n";
        return 1;
    }

    ChartConfig grouped_cfg = cfg;
    grouped_cfg.height = 420;
    grouped_cfg.title = "Grouped Bars by Month";
    const std::vector<double> grouped_a = {55, 80, 50, 65, 70, 55};
    const std::vector<double> grouped_b = {84, 98, 90, 86, 80, 73};
    if (!create_grouped_bar_chart(
            "examples/charts/grouped_bar_chart.svg", months, grouped_a, grouped_b, grouped_cfg)) {
        std::cerr << "grouped bar chart failed\n";
        return 1;
    }

    ChartConfig progress_cfg = cfg;
    progress_cfg.height = 280;
    progress_cfg.title = "Progress Bars";
    if (!create_horizontal_progress_chart(
            "examples/charts/horizontal_progress_chart.svg", {"Pipeline A", "Pipeline B"}, {62, 41}, progress_cfg)) {
        std::cerr << "horizontal progress chart failed\n";
        return 1;
    }

    ChartConfig ranked_cfg = cfg;
    ranked_cfg.height = 500;
    ranked_cfg.title = "Group Ranking";
    const std::vector<std::string> rank_labels = {"Group 1", "Group 2", "Group 3", "Group 4", "Group 5", "Group 6"};
    const std::vector<double> rank_values = {100, 78, 72, 56, 43, 22};
    if (!create_horizontal_ranked_bar_chart(
            "examples/charts/horizontal_ranked_bar_chart.svg", rank_labels, rank_values, ranked_cfg)) {
        std::cerr << "horizontal ranked bar chart failed\n";
        return 1;
    }

    ChartConfig showcase_cfg = cfg;
    showcase_cfg.title = "Theme Reference Panel";
    if (!create_theme_showcase("examples/charts/theme_showcase.svg", showcase_cfg)) {
        std::cerr << "theme showcase failed\n";
        return 1;
    }

    std::cout << "Generated charts in examples/charts/" << std::endl;
    return 0;
}
