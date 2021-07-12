library("ggplot2")
library("rcompanion")
library("ggpubr")

values_path = "../output/gmhi_values.csv"
output_path = "../output/violin.png"

df = read.csv(values_path)
# transposed = as.data.frame(t(df))
# reversed = rev(transposed)
# new = t(reversed)
# df = as.data.frame(new)
# head(df)
num_healthies = dim(df[df[, "isHealthy"] == 'True',])[1]
num_unhealthies = dim(df[df[, "isHealthy"] == 'False',])[1]
healthies = data.frame(df[df[, "isHealthy"] == 'True',])
unhealthies = data.frame(df[df[, "isHealthy"] == 'False',])
png(output_path)
    fig = ggplot(df, aes(x=isHealthy, y=gmhi_value, fill=isHealthy)) + 
        geom_violin(trim=FALSE) + 
        geom_boxplot(width=0.1, fill="white") +
        theme_classic()
    fig + 
        rremove("legend") + 
        theme(axis.text=element_text(size=14,face="bold"),
            axis.title=element_text(size=14,face="bold")) +
        scale_colour_manual(
            values=c("Healthy"="steelblue","Nonhealthy"="orange2")) +
        #stat_compare_means(
        #    label = "p",method = "wilcox.test",label.x.npc = "middle") +
            scale_fill_manual(values=c('steelblue','orange2'))+
            labs(x = "",y="GMHI 2.0")
dev.off()
system(paste("open ", output_path))
df[df[, "isHealthy"] == "True",][, 'isHealthy'] = 0
df[df[, "isHealthy"] == "False",][, 'isHealthy'] = 1
print(cliffDelta(data = df,gmhi_value ~ isHealthy))
print(wilcox.test(healthies$gmhi_value, unhealthies$gmhi_value))
